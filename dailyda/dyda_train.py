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


# Load the BertForSequenceClassification model, which will train both the embedding space and also the classifier layer on top
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 4,
    output_attentions = False,
    output_hidden_states = False,
)

# Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr = 5e-5,
                              eps = 1e-08
                              )

criterion = torch.nn.CrossEntropyLoss()

# Train on the gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def logits_to_preds(logits):
    max_indices = torch.argmax(logits, dim=1)

    # Create a one-hot tensor
    one_hot = torch.zeros(logits.size())
    one_hot.scatter_(1, max_indices.view(-1, 1), 1)

    return one_hot



epochs = 1
for _ in range(epochs):
    
    # ========== Training ==========
    
    # =============== TRAIN ================
    # Set model to training mode
    model.train()
    
    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    for step, batch in enumerate(train_loader):
        print(f"epoch {_ + 1}, batch {step}")
        b_input_ids = batch['ids']
        b_input_mask = batch['mask']
        b_token_type_ids = batch['token_type_ids']
        b_labels = batch['target']

        optimizer.zero_grad()

        # Forward pass
        train_output = model(b_input_ids, 
                            token_type_ids = b_token_type_ids, 
                            attention_mask = b_input_mask)

        # Backward pass
        loss = criterion(train_output.logits.to(torch.float), b_labels.to(torch.float))
        loss.backward()
        optimizer.step()

        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1


print("saving model")
model.save_pretrained("../../models/dyda.model")