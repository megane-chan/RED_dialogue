from openai import OpenAI
import pandas as pd
import re
import pickle
import numpy as np
import time
from dotenv import load_dotenv
from datasets import load_dataset, Dataset


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
        'c': 'commissive',
        'd': 'directive',
        'q': 'question',
        'questioning': 'question',
        'informative': 'inform',
        'i': 'inform'
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
                "content": "I will give you a sequence of utterances in the form:\n\n'''\n1. first utterance\n2. second utterance\n3. third utterance\nn. nth utterance\n'''\n\nPlease return the most accurate classification tag for all n utterances in the sequence in the following list format:\n\n'''\nClassifications:\n[classification for 1st utterance, classification for 2nd utterance, classification for 3rd utterance, classification for nth utterance]\n'''\n\nAdditionally, please return an explanation of your classification process for each utterance in the following format:\n\n'''\nExplanations:\n1. explanation for first classification\n2. explanation for second classification\n3. explanation for third classification\nn. explanation for nth classification\n'''"
                },
                {
                "role": "user",
                "content": user_prompt
                }

            ],
            #hyperparams, these seemed to work decently well, but can be experimented with
            temperature=0.9,
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

    ds = load_dataset("silicone", "dyda_da")
    
    #list of first 500 dyda utterances
    small_dyda = ds['train'][0:500]["Utterance"]

    system = "You are a speech act classifier that uses its expert knowledge of linguistics and speech act theory to make speech act classifications for a sequence of utterances. Utterances may be classified as one of the four following speech acts: Commisive, Directive, Informative, and Question.\n\n1. Commissive\nA commissive speech act is a type of speech act in linguistic pragmatics where the speaker commits to a future course of action or a particular state of affairs. In other words, it involves the speaker making a promise, commitment, or pledge to do something in the future. Commissive speech acts are performative in nature, meaning that the utterance of the words itself brings about the commitment or promise.\n\nKey features of commissive speech acts include:\n\nVerbs of Commitment or Promise: Commissive speech acts often involve the use of specific verbs that indicate commitment or promise, such as \"promise,\" \"commit,\" \"swear,\" \"guarantee,\" or \"pledge.\"\n\nFuture-Oriented Language: The speech act typically includes language expressing a commitment or intention to perform a future action. Phrases using future tenses like \"I will,\" \"I shall,\" or \"I am going to\" are common.\n\nFirst-Person Pronouns: Commissive speech acts are often expressed in the first person, using pronouns like \"I\" or \"we,\" as the speaker is personally committing to an action or promise.\n\nExplicit Declarations: The commitment or promise is usually explicitly stated, making the speech act clear and direct. The speaker may make an explicit declaration of their intention.\n\nPerformative Nature: Uttering the words of a commissive speech act is performative, meaning that saying the words brings about the commitment itself. For example, saying \"I promise\" is an act of making a promise.\n\nAssurance and Certainty: The speaker may use words conveying assurance or certainty to emphasize the commitment. Terms like \"assure,\" \"guarantee,\" or \"undertake\" may be present.\n\nExamples of commissive speech acts include statements like \"I promise to help you,\" \"I am committed to finishing the project on time,\" or \"I guarantee that I will be there.\" These speech acts play a significant role in interpersonal communication by expressing the speaker's commitment to future actions or obligations.\n\n2. Directive\nA directive speech act is a type of speech act in linguistic pragmatics where the speaker attempts to get the listener to do something. It involves the speaker influencing the behavior of the listener by making a request, giving a command, or offering an invitation. Directive speech acts are communicative acts through which the speaker aims to bring about a specific action or response from the listener.\n\nKey characteristics of directive speech acts include:\n\nImperative Verbs: These are verbs that convey commands or requests, such as \"do,\" \"please,\" \"stop,\" \"give,\" etc. Imperative sentences are often used to express directive speech acts.\n\nRequests: The speaker may use polite or direct language to ask the listener to perform a certain action. Phrases like \"Could you,\" \"Would you,\" or \"Please\" are common in directive speech acts.\n\nCommanding Language: Direct and explicit language is often used to issue commands. Imperative sentences, which lack a subject and begin with a verb, are a typical form of commanding language.\n\nPoliteness Strategies: Depending on cultural and social norms, speakers may employ politeness strategies when making requests or giving directives. This can include using softening expressions or framing requests in a more polite manner.\n\nQuestions as Requests: Sometimes, questions can be used as a subtle form of requesting action. For example, \"Can you pass the salt?\" is a request disguised as a question.\n\nContextual Clues: Understanding the context of the conversation is crucial for correctly interpreting directive speech acts. The same words or phrases can have different implications based on the context in which they are used.\n\nExamples of directive speech acts include requests like \"Please close the door,\" commands like \"Stop talking,\" and invitations like \"Join us for dinner.\" The effectiveness of a directive speech act depends on the listener's willingness to comply and the social dynamics between the speaker and the listener.\n\n\n3. Informative:\nAn informative speech act is a type of speech act in linguistic pragmatics where the speaker provides information or conveys knowledge to the listener. The primary goal of an informative speech act is to share facts, describe events, explain concepts, or communicate any form of information to the audience. Unlike other speech acts that may involve making promises, requests, or commands, informative speech acts focus on the transmission of information.\n\nKey characteristics of informative speech acts include:\n\nDeclarative Statements: Informative speech acts often take the form of declarative statements, where the speaker asserts a fact or provides information without necessarily eliciting a specific response from the listener.\n\nDescriptive Language: The language used in informative speech acts is typically descriptive, aiming to provide a clear and accurate portrayal of a subject, event, or concept.\n\nObjective Tone: The speaker adopts an objective and neutral tone when presenting information. The goal is to convey facts without expressing personal opinions or emotions.\n\nExplanations and Clarifications: Informative speech acts may involve explanations or clarifications to ensure that the information is understood by the audience. This could include defining terms, offering examples, or providing additional context.\n\nUse of Evidence: Speakers may support their informative speech acts by presenting evidence, data, or examples to enhance the credibility and reliability of the information being shared.\n\nEducational Intent: Informative speech acts are often employed in educational or instructional contexts where the speaker aims to enhance the listener's knowledge on a particular subject.\n\nExamples of informative speech acts include:\n\n\"The capital of France is Paris.\"\n\"Water boils at 100 degrees Celsius at sea level.\"\n\"The Industrial Revolution began in the late 18th century.\"\nIn everyday communication, informative speech acts are prevalent in educational settings, presentations, news reporting, and any situation where the primary purpose is to convey information rather than to perform an action or express emotions.\n\n4. Question\nA question speech act is a type of speech act in linguistic pragmatics where the speaker seeks information, clarification, or confirmation from the listener. The primary purpose of a question is to elicit a response or a specific piece of information from the person being addressed. Questions can take various forms and serve different communicative functions.\n\nKey characteristics of question speech acts include:\n\nInterrogative Form: Questions are typically formed using interrogative words, phrases, or structures. These can include question words like \"who,\" \"what,\" \"where,\" \"when,\" \"why,\" and \"how,\" as well as inversion of subject and verb.\n\nSeeking Information: The main goal of a question is to obtain information that the speaker does not currently possess. This information could relate to facts, opinions, experiences, or any other relevant details.\n\nOpen-Ended or Closed-Ended: Questions can be classified as open-ended or closed-ended. Open-ended questions allow for a more detailed and expansive response, while closed-ended questions usually have a limited set of possible answers and often require a simple \"yes\" or \"no.\"\n\nRhetorical Questions: In some cases, speakers may pose rhetorical questions, which are not meant to elicit a direct answer but rather to emphasize a point, express a statement, or engage the listener's thought.\n\nIntonation and Tone: The intonation and tone used when asking a question can convey additional information about the speaker's intent. For example, rising intonation at the end of a statement can sometimes indicate a question.\n\nContextual Relevance: The interpretation of a question often depends on the broader context of the conversation. Understanding the context helps in providing appropriate and relevant answers.\n\nExamples of question speech acts include:\n\n\"What is your name?\"\n\"Where did you go last night?\"\n\"How does this machine work?\"\n\"Did you enjoy the movie?\"\nQuestions play a crucial role in communication by facilitating the exchange of information, promoting dialogue, and engaging participants in conversation. They can be used in various social situations, such as interviews, casual conversations, educational settings, and problem-solving interactions.\n\nUse the following steps to make speech act classifications for utterances:\n\nStep 1: Identify the key linguistic and semantic features in the utterance.\nStep 2: Compare the key linguistic and semantic features of the utterance to the features of the dialogue acts.\nStep 3: Use this comparison to predict the dialogue act class that the utterance most likely belongs to"

    print("Starting dyda Classification")
    p, e = classify_with_gpt(small_dyda, system_instructions=system, num_trials=3, context_size=5)

    results = pd.DataFrame({'preds': p, 'explanations': e})
    results.to_csv('gpt_dyda_results.csv')