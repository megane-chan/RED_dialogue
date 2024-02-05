import pandas as pd
import numpy as np
import transformers
from transformers import BertForSequenceClassification
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#load all data

chats_df = pd.read_csv('data/chat.csv')
trans_df = pd.read_csv('data/trans.csv')



'''prepare dataset, for making classifications the model will be given the statement before the statement being predicted for additional context. For example, for the following sequence:

Everything is ON. I have everything.
I have nothing. Wait, just a second. Uno momento por favor.
Nothing [UI]

To make a prediction for the first statement, the input to the model will be:

"Everything is ON. I have everything."

For the second:

"Everything is ON. I have everything. I have nothing. Wait, just a second. Uno momento por favor"

And for the third:

"I have nothing. Wait, just a second. Uno momento por favor. Nothing [UI]"

'''
class BertDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        super(BertDataset, self).__init__()
        self.content=data['content']
        self.block=data['block']
        self.session=data['session']
        self.tokenizer=tokenizer
        self.max_length=max_length
        
    def __len__(self):
        return len(self.content)
    
    def __getitem__(self, index):
        
        if index > 0 and self.block[index] == self.block[index - 1] and self.session[index] == self.session[index - 1]:
            text1 = self.content[index - 1]
            text2 = self.content[index]
            inp = "[CLS] " + text1 + "[SEP]" + text2
        else:
            text = self.content[index]
            inp = "[CLS] " + text + "[SEP]"

        inputs = self.tokenizer.encode_plus(
            inp,
            add_special_tokens=False,
            return_attention_mask=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )

        ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        mask = inputs["attention_mask"]



        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
            }
    




#et up model stuff and dataset
batch_size = 16

model = BertForSequenceClassification.from_pretrained("../models/mrda_backwards_context.model")
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

chat_dataset = BertDataset(chats_df, tokenizer, max_length=256)
chat_loader = DataLoader(chat_dataset, batch_size=batch_size)

trans_dataset = BertDataset(trans_df, tokenizer, max_length=256)
trans_loader = DataLoader(trans_dataset, batch_size=batch_size)


#helper
def logits_to_preds(logits):
    max_indices = torch.argmax(logits, dim=1)

    # Create a one-hot tensor
    one_hot = torch.zeros(logits.size())
    one_hot.scatter_(1, max_indices.view(-1, 1), 1)

    return one_hot


#make predictions for chat data
model.eval()
chat_preds = []
for i, batch in enumerate(chat_loader):

    b_input_ids = batch['ids']
    b_input_mask = batch['mask']
    b_token_type_ids = batch['token_type_ids']


    with torch.no_grad():
        # Forward pass
        eval_output = model(b_input_ids, 
                            token_type_ids = None,
                            attention_mask = b_input_mask)
    
    logits = logits_to_preds(eval_output.logits.to(torch.float))
    logits = logits.detach().cpu().numpy()


    chat_preds.append(logits)

    if i % 100 == 0:
        print(f"Starting batch {i} for chat")


#same for trans data
trans_preds = []
for i, batch in enumerate(trans_loader):

    b_input_ids = batch['ids']
    b_input_mask = batch['mask']
    b_token_type_ids = batch['token_type_ids']


    with torch.no_grad():
        # Forward pass
        eval_output = model(b_input_ids, 
                            token_type_ids = None,
                            attention_mask = b_input_mask)
    
    logits = logits_to_preds(eval_output.logits.to(torch.float))
    logits = logits.detach().cpu().numpy()


    trans_preds.append(logits)

    if i % 100 == 0:
        print(f"Starting batch {i} for trans")



#format predictions

unpacked_chat = []
unpacked_trans = []

m = {
    0 : 'statement',
    1 : 'disruption',
    2 : 'backchannel',
    3 : 'floor-grabber',
    4 : 'question'
}

for batch in chat_preds:
    for p in batch:
        i = np.argmax(p)
        unpacked_chat.append(m[i])

for batch in trans_preds:
    for p in batch:
        i = np.argmax(p)
        unpacked_trans.append(m[i])



chat_preds_df = pd.DataFrame(unpacked_chat, columns=['BC_Chat_Preds'])
trans_preds_df = pd.DataFrame(unpacked_trans, columns=['BC_Trans_Preds'])


#save results to new csv
chat_preds_df.to_csv('new_results/chat_preds.csv', index=False)
trans_preds_df.to_csv('new_results/trans_preds.csv', index=False)