#!/usr/bin/env python3


# imports
from datasets import load_dataset, Dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AutoTokenizer
import numpy as np
import pandas as pd
import numpy as np
from tqdm import trange
from tqdm import trange


# 5 dialog act labels
labels = ["s", "d", "b", "f", "q"]
label2id = {label:idx for idx, label in enumerate(labels)}

# load in NEK21 session 16 data
trans16 = pd.read_csv('fewshot_labels/trans16_milan_merged.csv', index_col="Unnamed: 0").reset_index(drop=True)
trans16['label'] = trans16['labels_h'].replace(label2id)
ds = Dataset.from_pandas(trans16)

# uncomment this to check for errors in the script, will train model with only 100 rows
# fast way to check for errors
# ds['train'] = Dataset.from_pandas(ds['train'].to_pandas()[1:100])
# ds['validation'] = Dataset.from_pandas(ds['validation'].to_pandas()[1:10])

# loads in tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# labels in the correct order
labels = ["statement", "declarative question", "backchannel", "follow-me", "question"]

# Encodes utterances and translates their dialog act labels in one-hot format
def dataprep(samples):
  encoding = tokenizer.encode_plus(samples['content'], add_special_tokens = True,
                        max_length = 64,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation=True,
                        padding="max_length"
                   )
  samples['input_ids'] = encoding['input_ids']
  samples['attention_masks'] = encoding['attention_mask']
  ls = np.zeros(len(labels))
  ls[int(samples['label'])] = 1
  samples['labels'] = ls

  return samples

# Creates encoded dataset and sets the format to pytorch
encoded = ds.map(dataprep)
encoded.set_format("torch")


# Metrics used to train model
# Taken from https://towardsdatascience.com/fine-tuning-bert-for-text-classification-54e7df642894
def b_tp(preds, labels):
  '''Returns True Positives (TP): count of correct predictions of actual class 1'''
  return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_fp(preds, labels):
  '''Returns False Positives (FP): count of wrong predictions of actual class 1'''
  return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_tn(preds, labels):
  '''Returns True Negatives (TN): count of correct predictions of actual class 0'''
  return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_fn(preds, labels):
  '''Returns False Negatives (FN): count of wrong predictions of actual class 0'''
  return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_metrics(preds, labels):
  '''
  Returns the following metrics:
    - accuracy    = (TP + TN) / N
    - precision   = TP / (TP + FP)
    - recall      = TP / (TP + FN)
    - specificity = TN / (TN + FP)
  '''
  preds = np.argmax(preds, axis = 1).flatten()
  labels = labels.flatten()
  tp = b_tp(preds, labels)
  tn = b_tn(preds, labels)
  fp = b_fp(preds, labels)
  fn = b_fn(preds, labels)
  b_accuracy = (tp + tn) / len(labels)
  b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
  b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
  b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
  return b_accuracy, b_precision, b_recall, b_specificity

# Load the BertForSequenceClassification model, which will train both the embedding space and also the classifier layer on top
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = len(labels),
    output_attentions = False,
    output_hidden_states = False,
)

# Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr = 5e-5,
                              eps = 1e-08
                              )

# Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1810.04805.pdf
batch_size = 16


# Train on the gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create train and validation datasets. 80/20 split

trans16 = encoded.to_pandas()
train_df = trans16.sample(frac=.8)
val_df = trans16.drop(train_df.index)

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)

train_ds.set_format("torch")
val_ds.set_format("torch")

# Create the dataloaders from the samples
train_dataloader = DataLoader(
            train_ds,
            sampler = RandomSampler(train_ds),
            batch_size = batch_size
        )
validation_dataloader = DataLoader(
            val_ds,
            sampler = RandomSampler(val_ds),
            batch_size = batch_size
        )

# 1 epoch b.c. fewshot learning
epochs = 1
for _ in trange(epochs, desc = 'Epoch'):
    
    # ========== Training ==========
    
    # =============== TRAIN ================
    # Set model to training mode
    model.train()
    
    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(train_dataloader):
        # reshapes tensors from size (x, 1, y) to (x,y)
        # creates the batch
        isx,_, isy =  batch['input_ids'].size()
        imx,_, imy =  batch['input_ids'].size()
        b_input_ids = batch['input_ids'].reshape(isx,isy)
        b_input_mask = batch['attention_masks'].reshape(imx,imy)
        b_labels = batch['labels']

        optimizer.zero_grad()

        # Forward pass
        train_output = model(b_input_ids, 
                            token_type_ids = None, 
                            attention_mask = b_input_mask, 
                             labels = b_labels)

        # Backward pass
        train_output.loss.backward()
        optimizer.step()

        # Update tracking variables
        tr_loss += train_output.loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    # ============== VALIDATE =================

    # Set model to evaluation mode
    model.eval()

    # Tracking variables 
    val_accuracy = []
    val_precision = []
    val_recall = []
    val_specificity = []

    for batch in validation_dataloader:
        isx,_, isy =  batch['input_ids'].size()
        imx,_, imy =  batch['input_ids'].size()
        b_input_ids = batch['input_ids'].reshape(isx,isy)
        b_input_mask = batch['attention_masks'].reshape(imx,imy)
        b_labels = batch['labels']
        with torch.no_grad():
          # Forward pass
          eval_output = model(b_input_ids, 
                              token_type_ids = None,
                              attention_mask = b_input_mask)
                            
        logits = eval_output.logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()

        # Calculate validation metrics
        b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
        val_accuracy.append(b_accuracy)
        # Update precision only when (tp + fp) !=0; ignore nan
        if b_precision != 'nan': val_precision.append(b_precision)
        # Update recall only when (tp + fn) !=0; ignore nan
        if b_recall != 'nan': val_recall.append(b_recall)
        # Update specificity only when (tn + fp) !=0; ignore nan
        if b_specificity != 'nan': val_specificity.append(b_specificity)

    print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
    print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
    print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
    print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
    print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')

# Save model
model.save_pretrained("models/model_mrda_v2_fewshot_t2.model")