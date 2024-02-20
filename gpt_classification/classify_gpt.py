from openai import OpenAI
import pandas as pd
import re
import pickle
import numpy as np
import time
from dotenv import load_dotenv


def format_utterances(utterances_df):

    '''
    utterances_df (dataframe or listof (str)): This should be a pandas dataframe or a list of strings where each element in the df or list is an utterance. Indexing utterances_df like utterances_df[index] should return a string of an utterance

    Returns a string in the form:

    1. first utterance\n
    2. second utterance\n
    3. third utterance\n
    n. nth utterance

    Containing each utterance in utterances_df

    '''

    formatted_utterances = "\n".join([f"{i+1}. {utterance}" for i, utterance in enumerate(utterances_df)])

    return formatted_utterances



def extract_data(text, num_utterances):
    '''
    Extracts classifications and explanations from text in the format:

    Classifications:
    [<q>, <s>, <s>, <s>, <s>, <b>, <d>]


    Explanations:
    1. The utterance "What’s taking so long?" is a question because it starts with "What" and ends with a question mark.
    2. The utterance "You keep working." is a statement because it is a complete sentence that expresses a fact or opinion.
    3. The utterance "Eat something if you want." is a statement because it is a complete sentence that expresses a suggestion or possibility.
    4. The utterance "I see you are feeling bored. She is missing the letters. Read." is a statement because it is a sequence of complete sentences that express observations and instructions.
    5. The utterance "There’s a lot accumulated for you." is a statement because it is a complete sentence that expresses a fact or situation.
    6. The utterance "Right here." is a backchannel because it is a short utterance that indicates understanding or acknowledgement.
    7. The utterance "Oh, shit! Didn’t mean to do that." is an incomplete utterance because it is an interrupted utterance that is not complete.

    Regexes search for list formated data, and numbered explanations

    returns a list of classifications -> [<q>, <s>, <s>, <s>, <s>, <b>, <d>]
    returns a list of explanations -> [explanation 1,explanation 2, ...]


    '''



    # Extract Classifications
    pattern = re.compile(r'\[([^\]]*)\]')

    matches = pattern.findall(text)

    if matches:
        classifications = [value.strip() for value in matches[0].split(',') if value.strip()]

        if len(classifications) != num_utterances:
            classifications = ['Failed to retrieve'] * num_utterances
    else:
        classifications = ['Failed to retrieve'] * num_utterances

    # Extract Explanations
    pattern = re.compile(r'\d+\.\s(.+?)(?=\d+\.|\Z)', re.DOTALL)

    # Use findall to extract all matches
    explanations = pattern.findall(text)

    if explanations:
        # Strip '\n' characters from each statement
        explanations = [re.sub(r'\s+', ' ', statement) for statement in explanations]

        if len(explanations) != num_utterances:
            explanations = ['Failed to retrieve'] * num_utterances
    else:
        explanations = ['Failed to retrieve'] * num_utterances

    #make sure output is all the same format
    #sometimes gpt uses full names instead of tags
    s_map = {
        'question': 'q',
        'statement': 's',
        'follow me': 'f',
        'follow-me': 'f',
        'backchannel': 'b',
        'back channel': 'b',
        'back-channel': 'b',
        'incomplete utterance': 'd',
        'incomplete-utterance': 'd',
        'floor grabber': 'f',
        'floor-grabber': 'f' 
    }

    #further clean output
    special_characters = ['<', '>', "'", '"']

    classifications = ["".join(char for char in s if char not in special_characters).lower() for s in classifications]

    classifications = [s_map[s] if s in s_map else s for s in classifications]

    return classifications, explanations



def predict(utterances, num_utterances, system_instructions, max_retries=10):

    '''
    Utterances (str): Should be a sequence of utterances in the format:

    1. "first utterance"\n
    2. "second utterance"\n
    3. "third utterance"\n
    n. "nth utterance"

    This should be a single string

    Returns listof (str): This will be a list of classifications where each str in the list
    is a dialogue act prection for the corresponding utterance in utterances.

    '''

    user_prompt = utterances

    #retry api call up to ten times
    while max_retries > 0:
        max_retries -= 1

        try:

            response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                "role": "system",

                "content": system_instructions
                },
                {
                "role": "user",
                "content": "I will give you a sequence of utterances in the form:\n\n'''\n1. first utterance\n2. second utterance\n3. third utterance\nn. nth utterance\n'''\n\nPlease return the most accurate classification tag for all n utterances in the sequence in the following list format:\n\n'''\nClassifications:\n[<first classification tag>, <second classification tag>, <third classification tag>, <nth classification tag>]\n'''\n\nAdditionally, please return an explanation of your classification process for each utterance in the following format:\n\n'''\nExplanations:\n1. explanation for first classification\n2. explanation for second classification\n3. explanation for third classification\nn. explanation for nth classification\n'''"
                },
                {
                "role": "user",
                "content": user_prompt
                }

            ],
            #hyperparams, these seemed to work decently well, but can be experimented with
            temperature=0.38,
            max_tokens=700,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )

            #entire gpt response
            response_string = response.choices[0].message.content


            classifications, explanations = extract_data(response_string, num_utterances)

            return classifications, explanations
        
        except:
            #it the api call fails, try again after 15 seconds
            time.sleep(15)
            print(f"Retrying, {max_retries} tries left")

    #if every retry fails for this batch, give up
    return extract_data("this should yield two failed to retrieve lists", num_utterances)


def predict_all_data(data, context_size, system_instructions):

    '''
    Given a dataframe of utterances to classify, split up the data into groups of
    size context_size. This means each api call will include context_size number of utterances.
    
    data (dataframe or listof (str)): This should be a pandas dataframe or a list of strings where each element in the df or list is an utterance. Indexing utterances_df like utterances_df[index] should return a string of an utterance

    context_size (int): number of utterances that will be classified in single api call

    system_instructions (str): system message used in api call. See https://community.openai.com/t/the-role-of-system-prompts/149342/3 for more info on system prompts.
    

    '''


    #context size needs to be greater than or equal to the length of data
    if len(data) < context_size:
        raise Exception("context_size cannot be larger than lenght of data")

    #make preds for every utterance in dataset, giving gpt context_size at a time
    preds = []
    explanations = []

    for i in range((len(data) // context_size) + 1):
        start = context_size * i
        end = start + context_size

        
        if end > len(data):
            end = len(data)

        print(f"Classifying {start}:{end}")

        utterances = format_utterances(data[start:end])

        #wait some time between each api call
        time.sleep(5)

        p, e = predict(utterances, len(data[start:end]), system_instructions=system_instructions)


        preds = preds + p
        explanations = explanations + e

        if end == len(data):
            return preds, explanations
    
    return preds, explanations
    


def classify_with_gpt(data, context_size, system_instructions, num_trials=1):

    '''
    Uses gpt to make dialogue act classifications for every utterance in a dataset.

    data (dataframe or listof (str)): This should be a pandas dataframe or a list of strings where each element in the df or list is an utterance. Indexing utterances_df like utterances_df[index] should return a string of an utterance

    num_trials (int): The number of times gpt is used to make classifications. The mode classifications of all trials will be used as the final output. For example, if num_trials=3,
    the data will be ran through gpt 3 times, producing 3 predictions for each utterance. The mode of these 3 predictions will be used as the final prediction.

    system_instructions (str): system message used in api call. See https://community.openai.com/t/the-role-of-system-prompts/149342/3 for more info on system prompts.

    context_size (int): number of utterances that will be classified in single api call

    Returns:

    classifications (list of str): A list of classifictions for each utterance in data
    explanations (list of str): A list of explanations for each classification
    
    '''


    #we are going to make classifications for each utterance in our data num_trials times and take the mode classification for each as the final classification. This is to try an deal with gpt giving different answers sometimes.

    trial_outputs = []
  
    for i in range(num_trials):
        p, e = predict_all_data(data, context_size=context_size, system_instructions=system_instructions)
        trial_outputs.append((p,e))


    #after all trials of preds are done, get the mode of each one. I'm only going to save one of the explanations, just by whichever explanation with a corresponsing mode class comes first

    #ensure all are the same length. This should be handled in helper functions, so if things aren't the same size here, something weird is going on.
    true_length = len(data)
    for o in trial_outputs:
        if len(o[0]) != true_length:
            #save output into file for investigation
            with open("trial_outputs.pkl", "wb") as pickle_file:
                pickle.dump(trial_outputs, pickle_file)

            raise Exception(f"Size of preds list does not match size of labels: {len(o[0])} != {true_length}")
        
        if len(o[1]) != true_length:
            #save output into file for investigation
            with open("trial_outputs.pkl", "wb") as pickle_file:
                pickle.dump(trial_outputs, pickle_file)

            raise Exception(f"Size of examples list does not match size of labels: {len(o[1])} != {true_length}")
    
    
    #if only one trial was ran, modes dont have to be found
    if num_trials == 1:
        return trial_outputs[0][0], trial_outputs[0][1]

    #if everything is of the same size, get the mode class for each utterance

    pred_lists = [o[0] for o in trial_outputs]
    exp_lists = [o[1] for o in trial_outputs]

    final_preds = []
    final_exps = []

    for i in range(len(pred_lists[0])):


        curr_preds = [l[i] for l in pred_lists]
        curr_exps = [l[i] for l in exp_lists]

        # Find unique elements and their counts
        unique_elements, counts = np.unique(curr_preds, return_counts=True)

        # Find the first index where the mode occurs
        mode_index = np.where(counts == np.max(counts))[0][0]

        #get the mode pred and corresponding explanation
        mode_pred = curr_preds[mode_index]
        mode_exp = curr_exps[mode_index]

        final_preds.append(mode_pred)
        final_exps.append(mode_exp)

    return final_preds, final_exps








if __name__ == "__main__":
    #uses api key in .env
    load_dotenv()
    client = OpenAI()

    mrda = pd.read_csv('../mrda/MRDA_data.csv')
    small = mrda['utterance'][0:500]

    floor_grabber_system = "You are a dialogue act classifier that uses its expert knowledge of linguistics and dialogue act theory to predict dialogue act classifications for a sequence of utterances. \n\nUtterances may be classified as one of the five following dialogue acts:\n\nStatement <s>:\nThe <s> tag is the most widely used. Unless an utterance is completely indecipherable or else can be further described by a general tag as being a\ntype of question, backchannel, follow-me, or disruption, then its default status as a statement remains. \n\nFloor Grabber <f>:\nThe <f> tag is used to denote  Floor Grabber utterance. Floor grabbers usually mark instances in which a speaker has not been speaking and wants to gain the floor so that he may commence speaking. They are often repeated by the speaker to gain attention and are used by speakers to interrupt the current speaker who has the floor. Most often, floor grabbers tend to occur at the beginning of a speaker's turn.\nIn some cases, none of the speakers will have the floor, resulting in multiple speakers vying for the floor and consequently using floor grabbers to attain it. During such occurrences, many speakers talk over one another without actually having the floor.\n\nFloor grabbers are also used to mark instances in which a speaker who has the floor begins losing energy during his turn and then uses a floor grabber to either regain the attention of his audience or else because it seems as though he is relinquishing the\nfloor, which he does not wish to do. Such mid-speech floor grabbers are usually followed by a change in topic.\nFloor grabbers are generally louder than the surrounding speech. Although the energy of a floor grabber is relative to the energy of the surrounding speech, it is also relative to the energy of a speaker's normal speech.\n\nCommon floor grabbers include, but are not limited to, the following: \"well,\" \"and,\" \"but,\" \"so,\" \"um,\" \"uh,\" \"I mean,\" \"okay,\" and \"yeah.\" It is worth mentioning that the identification of floor grabbers is not merely based purely on the vocabulary used, but\nrather on the speaker's actual attempt, whether successful or not, to gain the floor. \n\nQuestions <q>:\nThe <q> tag is used to denote any form of question. Some examples of questions are: \"How are you?\", \"Did you get that thing?\", \"Who was that?\".\n\nIncomplete Utterance <d>:\nThe <d> tag is used to mark utterances that are abandoned, or incomplete.  Incomplete Utterances can be represented in an utterance through certain symbols. For example, an ellipsis can be used to mark an utterance as interrupted as in the following example: \" No, we already…\". Additionally, the \"[UI]\" tag can be used to indicate an unintelligible utterance as in the example: \"It’s just [UI]\".  Some examples of Incomplete Utterances are: \"Did you...\", \"I want [UI]\", \"There is a...\".\n\nBackchannel <b>:\nThe <b> tag marks utterances that are backchannels. Utterances that function as backchannels are not made by the speaker who has the\nfloor. Instead, backchannels are utterances made in the background that simply indicate that a listener is following along or at least is yielding the illusion that he is paying attention. When uttering backchannels, a speaker is not speaking directly to\nanyone in particular or even to anyone at all. Some examples of backchannels are: \"yeah\", \"OK\", \"uh-huh\", \"hmm\", \"right\", and \"I see\".\n\nUse the following steps to make dialogue act classifications for utterances:\n\nStep 1: Identify the key linguistic and semantic features in the utterance.\n\nStep 2: Compare the key linguistic and semantic features of the utterance to the features of the dialogue acts.\n\nStep 3: Use this comparison to predict the dialogue act class that the utterance most likely belongs to\n"

    print("Starting MRDA Classification")
    p, e = classify_with_gpt(small, system_instructions=floor_grabber_system, num_trials=3, context_size=5)

    results = pd.DataFrame({'preds': p, 'explanations': e})
    results.to_csv('mrda_results.csv')