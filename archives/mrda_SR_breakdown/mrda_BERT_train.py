#!/usr/bin/env python3
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


ds = load_dataset("silicone", "mrda")

# dry run with less data to check for errors
ds['train'] = Dataset.from_pandas(ds['train'].to_pandas()[1:100])
ds['validation'] = Dataset.from_pandas(ds['validation'].to_pandas()[1:10])

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

s_labels = ['statement', 'other']
s_id2label = {idx:label for idx, label in enumerate(s_labels)}
s_label2id = {label:idx for idx, label in enumerate(s_labels)}

labels = ["declarative question", "backchannel", "follow-me", "question"]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

# Encodes utterances and assigns them classified act labels
def dataprep(samples):
  encoding = tokenizer.encode_plus(samples['Utterance'], add_special_tokens = True,
                        max_length = 64,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation=True,
                        padding="max_length"
                   )
  samples['input_ids'] = encoding['input_ids']
  samples['attention_masks'] = encoding['attention_mask']
  if(samples['Label'] == 0):
     samples['is_statement'] = 1
  else:
    samples['is_statement'] = 0
    ls = np.zeros(len(labels))
    ls[samples['Label']-1] = 1
    samples['labels'] = ls

  return samples

# Creates encoded dataset and sets the format to pytorch
encoded = ds.map(dataprep)
encoded.set_format("torch")


#https://towardsdatascience.com/fine-tuning-bert-for-text-classification-54e7df642894

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

# Load the BertForSequenceClassification model, has 7 acts to classify on
s_model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = len(s_labels),
    output_attentions = False,
    output_hidden_states = False,
)

# Load the BertForSequenceClassification model, has 7 acts to classify on
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = len(labels),
    output_attentions = False,
    output_hidden_states = False,
)

# Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
optimizer = torch.optim.AdamW(s_model.parameters(), 
                              lr = 5e-5,
                              eps = 1e-08
                              )

# Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1810.04805.pdf
batch_size = 16


# Train on the gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

s_encoded_train = encoded['train'].to_pandas()['Label']

# Train and validation sets
train_dataloader = DataLoader(
            encoded['train'],
            sampler = RandomSampler(encoded['train']),
            batch_size = batch_size
        )

validation_dataloader = DataLoader(
            encoded['validation'],
            sampler = SequentialSampler(encoded['validation']),
            batch_size = batch_size
        )

# Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
epochs = 3
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

model.save_pretrained("models/model_mrda_v2_t1.model")
# Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr = 5e-5,
                              eps = 1e-08
                              )

# Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1810.04805.pdf
batch_size = 16


# Train on the gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train and validation sets
train_dataloader = DataLoader(
            encoded['train'],
            sampler = RandomSampler(encoded['train']),
            batch_size = batch_size
        )

validation_dataloader = DataLoader(
            encoded['validation'],
            sampler = SequentialSampler(encoded['validation']),
            batch_size = batch_size
        )

# Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
epochs = 3
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

model.save_pretrained("models/model_mrda_v2_t1.model")