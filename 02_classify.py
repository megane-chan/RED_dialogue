# %%
from datasets import Dataset
import pandas as pd
from transformers import BertForSequenceClassification, AutoTokenizer
from torch import tensor

# %%
#load model
model = BertForSequenceClassification.from_pretrained("models/model_mrda_v2_fewshot_t1.model/")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

labels = ["statement", "disruption", "backchannel", "follow-me", "question"]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

# %%

# Preproccesses data before classifcation
def preproccess(samples):
    encoding = tokenizer.encode_plus(samples['content'], add_special_tokens = True,
                        max_length = 32,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                        truncation=True,
                        padding="max_length")
        
    samples['input_ids'] = encoding['input_ids']
    samples['token_type_ids'] = encoding['token_type_ids']
    samples['attention_mask'] = encoding['attention_mask']
    return samples


# %%
# Uses trained model to classify
def classify(samples):
    out = model(samples['input_ids'], token_type_ids=samples['token_type_ids'], attention_mask=samples['attention_mask'])
    logits = out.logits.detach().cpu().numpy()

    samples['logits'] = logits[0]
    samples['labels_h'] = labels[logits.argmax()]
    samples['labels'] = logits.argmax()
    return samples

# %%
# Load in chat & transcript data
chats_df = pd.read_csv('data/chat.csv')
trans_df = pd.read_csv('data/trans.csv')

chats = Dataset.from_pandas(chats_df)
trans = Dataset.from_pandas(trans_df)

# %%
chats = chats.map(preproccess)
chats.set_format('torch')
chats = chats.map(classify)

chats.to_csv('results/chat_results.csv')

# %%
#classify transcripts
trans = trans.map(preproccess)
trans.set_format('torch')
trans = trans.map(classify)

trans.to_csv('results/trans_results.csv')


