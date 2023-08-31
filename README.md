# RED_dialogue

This project looks all the possible 



### Session Coding:

NEK19 Sessions: 19XX, where XX is the session number. For example, NEK19 session 3 will be coded as Session 1903
NEK21 Sessions: 21XX, where XX is the session number. For example, NEK21 session 9 will be coded as Session 2109
MAG21 Transcripts: Coded as 121XX where XX is the session number. For example, MAG21 Session 7 is coded as 12107

## files:

### Analysis:
01_parse.ipynb: reads in all the chat and transcript data for all sessions
02_classify.ipynb/.py: classifies both chat and transcript data for all sessions
03_bert_validation.ipynb: looks at NEK21 Session 16, which 2 humans have manually coded. looks at coding agreement between humans and BERT
04_markov.ipynb: uses markov chains to calculate the transition matrix probabilities. Contains visuals that look at high vs low performance teams and the significance of transition matrix values and its relation to performance
05_linreg_sigtest.ipynb: an alternative methodology to look at the significance of transition matrix probabilities and its relation to performance 

### BERT Model Training:

mrda/01_mrda_BERT_train.py: fine-tunes BERT based on MRDA dataset
mrda/02_fewshot_train.py: uses human coded labels to further fine-tune model
mrda/03_model_test.ipynb: tests accurary of the classification layer on the model created in mrda/02_fewshot_train.py using the MRDA test split

### Models

models/model_mrda_v2_fewshot_t1.model: Model used in all of analysis. Fine-tuned with MRDA, further tuned with NEK21 session 16 milan's labels
models/model_mrda_v2_t1.model: Model fine-tuned with MRDA, model_mrda_v2_fewshot_t1.model is based on this model

### Fewshot labeling

fewshot_labels/chat16.csv: The content of Session 2116's chats
fewshot_labels/trans16.csv: The content of Session 2116's transcripts
fewshot_labels/trans16_milan.csv: A CSV with just the labels that milan manually classified (not the actual utterances)
fewshot_labels/trans16_milan_merged.csv: A CSV with both the labels milan manually classified and the actual utterances

### Data

data/chat.csv: A CSV file containing all the chats in all sessions
data/trans.csv: A CSV file containing all the dialog utterances in all sessions

data/chats/NEK19/NEK19_X.txt: a file containing the raw chatlogs for NEK19 Session X
data/chats/NEK19/NEK19_X.txt: a file containing the raw chatlogs for NEK19 Session X
data/transcripts/NEK19/X.txt: a file containing the raw transcripts for NEK19 Session X
data/transcripts/NEK21/XX.txt: a file containing the raw transcripts for NEK21 Session XX
data/transcripts/NEK21_MAG/XX.txt: a file containing the raw transcripts for MAG21 Session XX