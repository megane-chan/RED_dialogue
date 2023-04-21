
# %% [markdown]
# <a href="https://colab.research.google.com/github/megane-chan/RED_dialogue/blob/main/swda_BERT_v1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
#!pip install --user -q transformers datasets torch scikit-learn tabulate tqdm

# %%
from datasets import load_dataset, Dataset
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from tabulate import tabulate
from tqdm import trange
import random
from transformers import DataCollatorWithPadding


ds = load_dataset("swda", "train")

# ds['train'] = Dataset.from_pandas(ds['train'].to_pandas()[1:100])
# ds['validation'] = Dataset.from_pandas(ds['validation'].to_pandas()[1:10])



# %%
# exploratory
ds['train']

# %%
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
label_names =  ["dummy", "state", "inform", "validate", "reject", "inquire", "direct"]
labels = label_names
id2label = {idx:label for idx, label in enumerate(label_names)}
label2id = {label:idx for idx, label in enumerate(label_names)}

# %%
RAW_ACT_TAGS = [ 'ad', 'qo', 'qy', 'arp_nd', 'sd', 'h', 'bh', 'no', '^2', '^g', 'ar', 'aa', 'sv', 'bk', 'fp', 'qw', 'b', 'ba', 't1', 'oo_co_cc', '+', 'ny', 'qw^d', 'x', 'qh', 'fc', 'fo_o_fw_"_by_bc', 'aap_am', '%', 'bf', 't3', 'nn', 'bd', 'ng', '^q', 'br', 'qy^d', 'fa', '^h', 'b^m', 'ft', 'qrr', 'na', ]
ACT_LABELS = { 'sd': 1, 'b': 3, 'sv': 1, 'aa': 3, '%': 0, 'ba': 3, 'qy': 5, 'x': 0, 'ny': 3, 'fc': 1, '%': 0, 'qw': 5, 'nn': 4, 'bk': 3, 'h': 5, 'qy^d': 5, 'fo_o_fw_"_by_bc': 0, 'bh': 5, '^q': 2, 'bf': 2, 'na': 3, 'ad': 6, '^2': 5, 'b^m': 3, 'qo': 5, 'qh': 1, '^h': 0, 'ar': 4, 'ng': 4, 'br': 4, 'no': 1, 'fp': 5, 'qrr': 5, 'arp_nd': 4, 't3': 6, 'oo_co_cc': 3, 't1': 0, 'bd': 0, 'aap_am': 3, '^g': 5, 'qw^d': 5, 'fa': 3, 'ft': 3, '+': 0}

# %%
def dataprep(samples):
  #print(ACT_LABELS[RAW_ACT_TAGS[samples['damsl_act_tag']]])
  encoding = tokenizer.encode_plus(samples['text'], add_special_tokens = True,
                        max_length = 32,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation=True,
                        padding="max_length"
                   )
  #print(encoding)
  samples['input_ids'] = encoding['input_ids']
  samples['attention_masks'] = encoding['attention_mask']
  ls = np.zeros(7)
  ls[ACT_LABELS[RAW_ACT_TAGS[samples['damsl_act_tag']]]] =1
  samples['labels'] = ls
  


  return samples
encoded = ds.map(dataprep)

#tokenizer.encode(train['text'][1], padding="max_length", truncation=True, max_length=128)

# %%
encoded.set_format("torch")

# %%
batch_size = 8
metric_name = "f1"

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           problem_type="single_label_classification", 
                                                           num_labels=len(label_names),
                                                           id2label=id2label,
                                                           label2id=label2id)

# %%
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    f"bert-finetuned-sem_eval-english",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    #push_to_hub=True,
)

# %%
#forward pass
#outputs = model(input_ids=encoded['train']['input_ids'][0].unsqueeze(0), labels=encoded['train'][0]['labels'].unsqueeze(0))
#outputs

# %%
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

# %%
# Load the BertForSequenceClassification model
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 7,
    output_attentions = False,
    output_hidden_states = False,
)

# Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr = 5e-5,
                              eps = 1e-08
                              )

# %%
model.train()

# %%
from sklearn.model_selection import train_test_split
import numpy as np
val_ratio = 0.2
# Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1810.04805.pdf
batch_size = 16


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Train and validation sets
train_set = encoded['train']

val_set = encoded['validation']
# Prepare DataLoader
train_dataloader = DataLoader(
            train_set,
            sampler = RandomSampler(train_set),
            batch_size = batch_size
        )

validation_dataloader = DataLoader(
            val_set,
            sampler = SequentialSampler(val_set),
            batch_size = batch_size
        )
# import evaluate

# accuracy = evaluate.load("accuracy")
# def compute_metrics(eval_pred):
#   predictions, labels = eval_pred
#   predictions = np.argmax(predictions, axis=1)
#   return accuracy.compute(predictions=predictions, references=labels)

# # Fine tuning
# training_args = TrainingArguments(

#     output_dir="model_v1",

#     learning_rate=2e-5,

#     per_device_train_batch_size=16,

#     per_device_eval_batch_size=16,

#     num_train_epochs=2,

#     weight_decay=0.01,

#     evaluation_strategy="epoch",

#     save_strategy="epoch",

#     load_best_model_at_end=True,

# )

# trainer =Trainer(model=model, 
#                  args=training_args, 
#                  train_dataset=ds['train'],
#                  eval_dataset=ds['validation'],
#                  tokenizer=tokenizer,
#                  data_collator=data_collator,
#                  compute_metrics=compute_metrics)

# trainer.train()

# %%
from tqdm import trange
# Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
epochs = 10
a = 0
for _ in trange(epochs, desc = 'Epoch'):
    
    # ========== Training ==========
    
    # Set model to training mode
    model.train()
    
    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    print("Train")
    for step, batch in enumerate(train_dataloader):
        #batch = tuple(t.to(device) for t in batch)
        a = batch
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


#    ============== VALIDATE =================

    # Set model to evaluation mode
    model.eval()

    # Tracking variables 
    val_accuracy = []
    val_precision = []
    val_recall = []
    val_specificity = []

    c = 0
    print("Validation")
    for batch in validation_dataloader:
        #batch = tuple(t.to(device) for t in batch)
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
        c += 1

    print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
    print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy)/len(val_accuracy)))
    print('\t - Validation Precision: {:.4f}'.format(sum(val_precision)/len(val_precision)) if len(val_precision)>0 else '\t - Validation Precision: NaN')
    print('\t - Validation Recall: {:.4f}'.format(sum(val_recall)/len(val_recall)) if len(val_recall)>0 else '\t - Validation Recall: NaN')
    print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity)/len(val_specificity)) if len(val_specificity)>0 else '\t - Validation Specificity: NaN')

# %%

# %%


model.save_pretrained("model__v1_t3.model")