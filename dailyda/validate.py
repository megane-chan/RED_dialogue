import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
import transformers
from transformers import BertForSequenceClassification
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt




ds = load_dataset("silicone", "dyda_da")




class BertDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        super(BertDataset, self).__init__()
        self.tokenizer=tokenizer
        self.data = data
        self.max_length=max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        

        utterance = self.data['Utterance'][index]

        inputs = self.tokenizer.encode_plus(
            utterance,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )

        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]

        label = np.zeros(4)
        label[self.data["Label"][index]] = 1

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(label, dtype=torch.long)
            }
    



batch_size = 256

tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = BertDataset(ds['train'], tokenizer, max_length=256)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = BertDataset(ds['validation'], tokenizer, max_length=256)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



def logits_to_preds(logits):
    max_indices = torch.argmax(logits, dim=1)

    # Create a one-hot tensor
    one_hot = torch.zeros(logits.size())
    one_hot.scatter_(1, max_indices.view(-1, 1), 1)

    return one_hot


model = BertForSequenceClassification.from_pretrained("../../models/dyda.model")

# ============== VALIDATE =================

# Set model to evaluation mode
model.eval()

preds = []
true = []

for i, batch in enumerate(test_loader):
    print(f'Batch: {i} ')
    b_input_ids = batch['ids']
    b_input_mask = batch['mask']
    b_token_type_ids = batch['token_type_ids']
    b_labels = batch['target']

    with torch.no_grad():
        # Forward pass
        eval_output = model(b_input_ids, 
                            token_type_ids = None,
                            attention_mask = b_input_mask)
    
    logits = logits_to_preds(eval_output.logits.to(torch.float))
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to(torch.float).cpu().numpy()

    preds.append(logits)
    true.append(label_ids)

unpacked_preds = []
unpacked_true = []

m = {
    0 : 'commissive',
    1 : 'directive',
    2 : 'inform',
    3 : 'question'
}

for batch in preds:
    for p in batch:
        i = np.argmax(p)
        unpacked_preds.append(m[i])

for batch in true:
    for p in batch:
        i = np.argmax(p)
        unpacked_true.append(m[i])

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


labels = ['commissive','directive','inform','question']

# Example lists of true labels and model predictions
true_labels = pd.DataFrame(unpacked_true)[0]
model_predictions = pd.DataFrame(unpacked_preds)[0]

# Convert string labels to numerical labels if needed (depends on the specific metrics you want to compute)
# For example, if using precision, recall, and F1-score, numerical labels are required
label_mapping = {label: idx for idx, label in enumerate(true_labels.unique())}
true_labels_numeric = true_labels.map(label_mapping)
model_predictions_numeric = model_predictions.map(label_mapping)

# Compute evaluation metrics
accuracy = accuracy_score(true_labels, model_predictions)
precision = precision_score(true_labels_numeric, model_predictions_numeric, average='weighted')
recall = recall_score(true_labels_numeric, model_predictions_numeric, average='weighted')
f1 = f1_score(true_labels_numeric, model_predictions_numeric, average='weighted')

# Display the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Display detailed classification report
print("\nClassification Report:")
print(classification_report(true_labels, model_predictions))

# Compute and display confusion matrix with columns
conf_matrix = confusion_matrix(true_labels, model_predictions, labels=true_labels.unique())
conf_matrix_df = pd.DataFrame(conf_matrix, index=true_labels.unique(), columns=true_labels.unique())
print("\nConfusion Matrix:")
print(conf_matrix_df)

# Calculate accuracy for each class
class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
class_accuracies_df = pd.DataFrame({'Accuracy': class_accuracies}, index=true_labels.unique())
print("\nAccuracy for Each Class:")
print(class_accuracies_df)

import matplotlib.pyplot as plt
import seaborn as sns


# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=true_labels.unique(), yticklabels=true_labels.unique())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('BERT trained on DYDA')
plt.savefig('BERTcfmx.png')