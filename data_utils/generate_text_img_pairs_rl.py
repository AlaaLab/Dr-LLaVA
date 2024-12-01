### generate the question answer pairs for running in the VQA model
### using VLLaVa model and data from the pathology images


### import the necessary packages
import os
import sys
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import pickle
import argparse
import shutil
import time
import glob
from tqdm import tqdm
from text_repo import  text_augmentation_repo



sys.path.append('/Users/ssun2/Documents/VLM/HemaVisionQA-dev/')
sys.path.append('/Users/ssun2/Documents/VLM/HemaVisionQA-dev/utils/')
sys.path.append('/Users/ssun2/Documents/VLM/HemaVisionQA-dev/utils/data_preprocessining/')
from utils import *
from data_preprocessing import *
from data_preprocessing.generate_text_img_pairs import *
from data_preprocessing.text_repo import  text_augmentation_repo

### this is how we want to store the data
'''
{'id': '000000033471', 
 'image': '000000033471.jpg', 
 'conversations': [{'from': 'human', 'value': '<image>\nWhat are the colors of the bus in the image?'}, {'from': 'gpt', 'value': 'The bus in the image is white and red.'}, 
                   {'from': 'human', 'value': 'What feature can be seen on the back of the bus?'}, {'from': 'gpt', 'value': 'The back of the bus features an advertisement.'}, 
                   {'from': 'human', 'value': 'Is the bus driving down the street or pulled off to the side?'}, {'from': 'gpt', 'value': 'The bus is driving down the street, which is crowded with people and other vehicles.'}]}
'''

def strip_skip_special_tokens(text, soa = '<PAD>', eoa= '<PAD>'):
    return text.replace(soa, '').replace(eoa, '')

### 
def hint_style_question(prior_answers, query):
    ### input is a string, the output is a query with this template
    ### Doctor believe <prior_answers>, <query>?

    return 'Doctor believe '+prior_answers+', '+query+'?'


def hint_style_answer(agreement, answer):
    ### input is a string, the output is a query with this template
    ### if agreement == True, then the answer just answer
    ### if agreement == False, then the answer is 'I don't agree with you. I think <answer>.'
    if agreement:
        return 'Yes, ' +answer
    else:
        return 'I don\'t agree with you. I think '+answer+'.'


def confirmation_style_question(prior_answers):
    ### input is a string, the output is a query with this template
    ### Doctor believe <prior_answers>, <query>?

    return 'Doctor believe '+prior_answers+', Do you agree?'

def confirmation_style_answer(agreement,  answer):
    ### input is a string, the output is a query with this template
    ### if agreement == True, then the answer just answer
    ### if agreement == False, then the answer is 'I don't agree with you. I think <answer>.'
    if agreement:
        return 'Yes, ' +answer
    else:
        return "I don\'t agree with you. I think "+answer


def create_df():
    ### this function is to create the dataframe
    ### dataframe columns: 'id', 'Plasma_cell', 'Myeloblast', 'Promyelocyte', 'Myelocyte', 'Metamyelocyte', 'Band_neutrophil', 'Segmented_neutrophil', 'Eosinophil', 'Basophil', 'Monocyte', 'Lymphocyte', 'Normoblast', 'Artifact', 'Mitotic_Body', 'quality', 'num_cells'
    ### the input is the list of files
    ### the output is the dataframe
    ### the dataframe will be used to create the json files
    ### the json files will be used for the VQA model
    img_dirs = glob.glob('../Data/ucsf_data/data_preselected/*/*')
    img_IDs = [img_dir.split('/')[-1].split('.')[0] for img_dir in img_dirs]
    print('The number of images is: ', len(img_IDs))
    print(img_dirs[:5])
    #raise NotImplementedError('The function is not implemented yet')
    ### making sure the image name are unique
    assert len(img_IDs) == len(set(img_IDs))
    #### also get the diagnosis information
    diags = [img_dir.split('/')[-2] for img_dir in img_dirs]
    ### create a dataframe to store the information
    df = pd.DataFrame({'id': img_IDs, 'diagnosis': diags})

    ### create all the other columns
    cell_types = ['Plasma_cell', 'Myeloblast', 'Promyelocyte', 'Myelocyte', 'Metamyelocyte', 'Band_neutrophil', 'Segmented_neutrophil', 'Eosinophil', 'Basophil', 'Monocyte', 'Lymphocyte', 'Normoblast', 'Artifact', 'Mitotic_Body', 'quality', 'num_cells']
    for cell_type in cell_types:
        df[cell_type] = np.nan
    

    ### if the diagnosis is not clot or blood, then we put the quality as 0, which means adequate
    df.loc[df['diagnosis'] == 'clot', 'quality'] = 2
    df.loc[df['diagnosis'] == 'blood', 'quality'] = 1

    ### all the others, starting with adequate, we put the quality as 0, which means adequate
    df.loc[df['diagnosis'].str.startswith('adequate'), 'quality'] = 0

    ### if the diagnosis is adequate_normal, then we put each cell as 1 and sum as the overall
    df.loc[df['diagnosis'] == 'adequate_normal', 'Plasma_cell'] = 1
    df.loc[df['diagnosis'] == 'adequate_normal', 'Myeloblast'] = 1
    df.loc[df['diagnosis'] == 'adequate_normal', 'Promyelocyte'] = 1
    df.loc[df['diagnosis'] == 'adequate_normal', 'Myelocyte'] = 1
    df.loc[df['diagnosis'] == 'adequate_normal', 'Metamyelocyte'] = 1
    df.loc[df['diagnosis'] == 'adequate_normal', 'Band_neutrophil'] = 1
    df.loc[df['diagnosis'] == 'adequate_normal', 'Segmented_neutrophil'] = 1
    df.loc[df['diagnosis'] == 'adequate_normal', 'Eosinophil'] = 1
    df.loc[df['diagnosis'] == 'adequate_normal', 'Basophil'] = 1
    df.loc[df['diagnosis'] == 'adequate_normal', 'Monocyte'] = 1
    df.loc[df['diagnosis'] == 'adequate_normal', 'Lymphocyte'] = 1
    df.loc[df['diagnosis'] == 'adequate_normal', 'Normoblast'] = 1
    df.loc[df['diagnosis'] == 'adequate_normal', 'Artifact'] = 1
    df.loc[df['diagnosis'] == 'adequate_normal', 'Mitotic_Body'] = 1
    df.loc[df['diagnosis'] == 'adequate_normal', 'num_cells'] = 14

    ### if the diagnosis is adequate_AML or adequate_AML_Mono, then we put each cell as 1 but the Myeloblast as 10 and sum as the overall
    df.loc[df['diagnosis'].isin(['adequate_AML','adequate_AML_Mono']), 'Plasma_cell'] = 1
    df.loc[df['diagnosis'].isin(['adequate_AML','adequate_AML_Mono']), 'Myeloblast'] = 10
    df.loc[df['diagnosis'].isin(['adequate_AML','adequate_AML_Mono']), 'Promyelocyte'] = 1
    df.loc[df['diagnosis'].isin(['adequate_AML','adequate_AML_Mono']), 'Myelocyte'] = 1
    df.loc[df['diagnosis'].isin(['adequate_AML','adequate_AML_Mono']), 'Metamyelocyte'] = 1
    df.loc[df['diagnosis'].isin(['adequate_AML','adequate_AML_Mono']), 'Band_neutrophil'] = 1
    df.loc[df['diagnosis'].isin(['adequate_AML','adequate_AML_Mono']), 'Segmented_neutrophil'] = 1
    df.loc[df['diagnosis'].isin(['adequate_AML','adequate_AML_Mono']), 'Eosinophil'] = 1
    df.loc[df['diagnosis'].isin(['adequate_AML','adequate_AML_Mono']), 'Basophil'] = 1
    df.loc[df['diagnosis'].isin(['adequate_AML','adequate_AML_Mono']), 'Monocyte'] = 1
    df.loc[df['diagnosis'].isin(['adequate_AML','adequate_AML_Mono']), 'Lymphocyte'] = 1
    df.loc[df['diagnosis'].isin(['adequate_AML','adequate_AML_Mono']), 'Normoblast'] = 1
    df.loc[df['diagnosis'].isin(['adequate_AML','adequate_AML_Mono']), 'Artifact'] = 1
    df.loc[df['diagnosis'].isin(['adequate_AML','adequate_AML_Mono']), 'Mitotic_Body'] = 1
    df.loc[df['diagnosis'].isin(['adequate_AML','adequate_AML_Mono']), 'num_cells'] = 23

    ### if the diagnosis is adequate_MM, then we put each cell as 1 but the Plasma_cell as 10 and sum as the overall
    df.loc[df['diagnosis'] == 'adequate_MM', 'Plasma_cell'] = 10
    df.loc[df['diagnosis'] == 'adequate_MM', 'Myeloblast'] = 1
    df.loc[df['diagnosis'] == 'adequate_MM', 'Promyelocyte'] = 1
    df.loc[df['diagnosis'] == 'adequate_MM', 'Myelocyte'] = 1
    df.loc[df['diagnosis'] == 'adequate_MM', 'Metamyelocyte'] = 1
    df.loc[df['diagnosis'] == 'adequate_MM', 'Band_neutrophil'] = 1
    df.loc[df['diagnosis'] == 'adequate_MM', 'Segmented_neutrophil'] = 1
    df.loc[df['diagnosis'] == 'adequate_MM', 'Eosinophil'] = 1
    df.loc[df['diagnosis'] == 'adequate_MM', 'Basophil'] = 1
    df.loc[df['diagnosis'] == 'adequate_MM', 'Monocyte'] = 1
    df.loc[df['diagnosis'] == 'adequate_MM', 'Lymphocyte'] = 1
    df.loc[df['diagnosis'] == 'adequate_MM', 'Normoblast'] = 1
    df.loc[df['diagnosis'] == 'adequate_MM', 'Artifact'] = 1
    df.loc[df['diagnosis'] == 'adequate_MM', 'Mitotic_Body'] = 1
    df.loc[df['diagnosis'] == 'adequate_MM', 'num_cells'] = 23  

    return df


def randomize(argument_threshold, permutated_threshold):
    # randimize a number between 0 and 1
    # if the number is less than argument_threshold, then return 0, which mean the answer is not argumented
    # randomly another value choose a number between 0 and 1
    # if the number is less than permutated_threshold, then return 1, which mean the answer is permutated
    
    rand = random.random()
    if rand < argument_threshold:
        argument = 0
    else:
        argument = 1
    rand = random.random()
    if rand < permutated_threshold:
        permutated = 1
    else:
        permutated = 0


def create_json_files(df, _dir, test = False):
    ### this function is to create the json files
    ### the input is the dataframe
    ### the output is the json files
    ### the json files will be used for the VQA model
    ### the json files will

    ### dataframe columns: 'id', 'Plasma_cell', 'Myeloblast', 'Promyelocyte', 'Myelocyte', 'Metamyelocyte', 'Band_neutrophil', 'Segmented_neutrophil', 'Eosinophil', 'Basophil', 'Monocyte', 'Lymphocyte', 'Normoblast', 'Artifact', 'Mitotic_Body', 'quality', 'num_cells'
    ### the json files will have the following keys: 'id', 'image', 'conversations'

    ### create a list to store the json files

    ### df is the input dataframe which we use to compose the json files

    question_answers_set = text_augmentation_repo()
    if test:
        question_answers_set.switch_split()
    json_files = []
    random_threshold = 0.1
    permutated_threshold = 0#.4
    ### loop through the dataframe

    for _index, i in enumerate(tqdm(range(len(df)))):   
        id = df.iloc[i]['id']
        image = df.iloc[i]['id'] + '.jpeg'
        conversations = []
        diagnosis = df.iloc[i]['diagnosis']

        ### use the logit flow to create the conversations
        while True:

            ### first question
            single_conversation = {}
            single_conversation['from'] = 'human'
            text_before = random.sample([0,1],1)[0]

            argument, permutated = 0,0 #randomize(random_threshold, permutated_threshold) # not doing any stupid things
            if argument == 1:
                raise NotImplementedError('The argument is not implemented yet')
                if text_before == 1:
                    single_conversation['value'] = confirmation_style_question() +'     \n'+'<image>' #'Tell me the quality of the pathology patches.'
                else:
                    assert text_before == 0, 'The text_before is not 0 or 1'
                    single_conversation['value'] = '<image>' +'\n'+ question_answers_set._sample('q_0') #'Tell me the quality of the pathology patches.'

            else:
                if text_before == 1:
                    single_conversation['value'] = question_answers_set._sample('q_0') +'     \n'+'<image>' #'Tell me the quality of the pathology patches.'
                else:
                    assert text_before == 0, 'The text_before is not 0 or 1'
                    single_conversation['value'] = '<image>' +'\n'+ question_answers_set._sample('q_0') #'Tell me the quality of the pathology patches.'
           
            conversations.append(single_conversation)

            if df.iloc[i]['quality'] == 0:
                assert diagnosis != 'blood', 'The diagnosis is blood, but the model said the quality is adequate.'
                assert diagnosis != 'clot', 'The diagnosis is clot, but the model said the quality is adequate.'
                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_01')  #'The quality of the pathology patches is adequate. It is good for the diagnosis.'
                conversations.append(single_conversation)
            elif df.iloc[i]['quality'] == 1:
                assert diagnosis == 'blood', 'The diagnosis is not blood, but the model said the quality is blood.'
                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_02') #'The quality of the pathology patches is blood. It is not good for the diagnosis.'
                conversations.append(single_conversation)
            elif df.iloc[i]['quality'] == 2:
                assert diagnosis == 'clot', 'The diagnosis is not clot, but the model said the quality is clot.'
                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_03')  #'The quality of the pathology patches is clot. It is not good for the diagnosis.'
                conversations.append(single_conversation)
            last_answer = strip_skip_special_tokens(single_conversation['value'])

            ### second question
            single_conversation = {}
            single_conversation['from'] = 'human'
            single_conversation['value'] =question_answers_set._sample('q_1') # question_answers_set._sample('Observation')+ ': '+last_answer +' '+ question_answers_set._sample('q_1')#'Does this slide looks quantifiable'
            conversations.append(single_conversation)

            if df.iloc[i]['quality'] == 0:
                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_11') #'Yes, this slide looks normal to me. Acceptable for quatification.'
                conversations.append(single_conversation)

                last_answer = strip_skip_special_tokens(single_conversation['value'])
            elif df.iloc[i]['quality'] == 1:
                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_12')#'No, this slide has a lot of red blood cells. Not acceptable for quatification.'
                conversations.append(single_conversation)

                last_answer = strip_skip_special_tokens(single_conversation['value'])

                single_conversation = {}
                single_conversation['from'] = 'human'
                single_conversation['value'] = question_answers_set._sample('q_2')# question_answers_set._sample('Observation')+': '+ \
                #last_answer +' ' + question_answers_set._sample('q_2')
                conversations.append(single_conversation)

                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_23')
                conversations.append(single_conversation)

                last_answer = strip_skip_special_tokens(single_conversation['value'])

                single_conversation = {}
                single_conversation['from'] = 'human'
                single_conversation['value'] = question_answers_set._sample('q_23') #question_answers_set._sample('Observation')+': '+ \
                #last_answer +' '+ question_answers_set._sample('q_23')
                conversations.append(single_conversation)

                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_231')
                conversations.append(single_conversation)

                last_answer = strip_skip_special_tokens(single_conversation['value'])

                single_conversation = {}
                single_conversation['from'] = 'human'
                single_conversation['value'] = question_answers_set._sample('q_231') #question_answers_set._sample('Observation')+': '+ \
                #last_answer +' '+question_answers_set._sample('q_231')
                conversations.append(single_conversation)

                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_2311')
                conversations.append(single_conversation)

                

                break
            elif df.iloc[i]['quality'] == 2:
                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_13') #'No, this slide has a lot of clot. Not acceptable for quatification.'
                conversations.append(single_conversation)

                last_answer = strip_skip_special_tokens(single_conversation['value'])

                single_conversation = {}
                single_conversation['from'] = 'human'
                single_conversation['value'] = question_answers_set._sample('q_2') #question_answers_set._sample('Observation')+ ': '+\
                #last_answer +' '+question_answers_set._sample('q_2')
                conversations.append(single_conversation)

                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_23')
                conversations.append(single_conversation)

                last_answer = strip_skip_special_tokens(single_conversation['value'])

                single_conversation = {}
                single_conversation['from'] = 'human'
                single_conversation['value'] =question_answers_set._sample('q_23') # question_answers_set._sample('Observation')+': '+ \
                #last_answer +' '+question_answers_set._sample('q_23')
                conversations.append(single_conversation)

                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_231')
                conversations.append(single_conversation)

                last_answer = strip_skip_special_tokens(single_conversation['value'])

                single_conversation = {}
                single_conversation['from'] = 'human'
                single_conversation['value'] = question_answers_set._sample('q_231') #question_answers_set._sample('Observation')+': '+ \
                #last_answer +' '+question_answers_set._sample('q_231')
                conversations.append(single_conversation)

                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_2311')
                conversations.append(single_conversation)

                
                break

            ### third question
            single_conversation = {}
            single_conversation['from'] = 'human'
            single_conversation['value'] =question_answers_set._sample('q_2') #question_answers_set._sample('Observation')+ ': '+\
                #last_answer +' '+ question_answers_set._sample('q_2') #'Now, let us look at the cells. Did you observe any diseases-related abnormality in the patch?'
            conversations.append(single_conversation)
            
            if df.iloc[i]['Myeloblast']/df.iloc[i]['num_cells']>0.2:
                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_21')  #'Yes, I saw the proliferation of certain celltypes, which could be the sign of potential hematological malignancy.'
                conversations.append(single_conversation)

                last_answer = strip_skip_special_tokens(single_conversation['value'])

                ### a follow up question
                single_conversation = {}
                single_conversation['from'] = 'human'
                single_conversation['value'] =question_answers_set._sample('q_21') #question_answers_set._sample('Observation')+ ': '+\
                #last_answer +' '+ question_answers_set._sample('q_21') #'What is the celltype that you saw has the abnormal proliferation?'
                conversations.append(single_conversation)

                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_211')#'I saw the proliferation of Myeloblast.'
                conversations.append(single_conversation)

                last_answer = strip_skip_special_tokens(single_conversation['value'])


                ### follow up with the final question about the diagnosis
                single_conversation = {}
                single_conversation['from'] = 'human'
                single_conversation['value'] = question_answers_set._sample('q_211')  #question_answers_set._sample('Observation')+': '+ \
                #last_answer +' '+question_answers_set._sample('q_211') #'What is potential disease that you saw?' 
                conversations.append(single_conversation)

                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_2111') #'I saw the proliferation of Myeloblast, which could be the sign of acute myeloid leukemia.'
                conversations.append(single_conversation)
                break

            elif df.iloc[i]['Plasma_cell']/df.iloc[i]['num_cells']>0.1:
                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_22') #'Yes, I saw the proliferation of certain celltypes, which could be the sign of potential hematological malignancy.'    
                conversations.append(single_conversation)

                last_answer = strip_skip_special_tokens(single_conversation['value'])

                ### a follow up question
                single_conversation = {}
                single_conversation['from'] = 'human'
                single_conversation['value'] = question_answers_set._sample('q_22')  #question_answers_set._sample('Observation')+': '+ \
                #last_answer +' '+question_answers_set._sample('q_22') #'What is the celltype that you saw has the abnormal proliferation?'``
                conversations.append(single_conversation)
                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_221') #'I saw the proliferation of Plasma cell.'
                conversations.append(single_conversation)

                last_answer = strip_skip_special_tokens(single_conversation['value'])

                ### follow up with the final question about the diagnosis


                single_conversation = {}
                single_conversation['from'] = 'human'
                single_conversation['value'] = question_answers_set._sample('q_221') #question_answers_set._sample('Observation')+': '+ \
                #last_answer +' '+question_answers_set._sample('q_221') #'What is potential disease that you saw?'
                conversations.append(single_conversation)

                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_2211') #'I saw the proliferation of Plasma cell, which could be the sign of multiple myeloma.'
                conversations.append(single_conversation)
                break

                

            else:
                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_24') #'No, I did not see any abnormality in the patch. The patch looks normal to me.'
                conversations.append(single_conversation)

                last_answer = strip_skip_special_tokens(single_conversation['value'])

                ### follow up with the final question about the diagnosis
                single_conversation = {}
                single_conversation['from'] = 'human'
                single_conversation['value'] =question_answers_set._sample('q_24') # question_answers_set._sample('Observation')+': '+ \
                #last_answer +' '+question_answers_set._sample('q_24') #'What is potential disease that you saw?'
                conversations.append(single_conversation)

                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_241') #'I did not see any abnormality in the patch. The patch looks normal to me.'
                conversations.append(single_conversation)

                last_answer = strip_skip_special_tokens(single_conversation['value'])

                single_conversation = {}
                single_conversation['from'] = 'human'
                single_conversation['value'] = question_answers_set._sample('q_241') #question_answers_set._sample('Observation')+': '+ \
                #last_answer +' '+question_answers_set._sample('q_241') #'What is potential disease that you saw?'
                conversations.append(single_conversation)

                single_conversation = {}
                single_conversation['from'] = 'gpt'
                single_conversation['value'] = question_answers_set._sample('a_2411') #'I did not see any abnormality in the patch. The patch looks normal to me.'
                conversations.append(single_conversation)

                assert diagnosis != 'blood', 'The diagnosis is blood, but the model did not see any abnormality in the patch. The patch looks normal to me.'
                assert diagnosis != 'clot', 'The diagnosis is clot, but the model did not see any abnormality in the patch. The patch looks normal to me.'
                break

                
            
        # add the data point to json file
        # the json files will have the following keys: 'id', 'image', 'conversations'

        # check the conversations has 'human' and 'gpt' equal
        
        assert len([conver['from'] for conver in conversations if conver['from'] == 'human']) == len([conver['from'] for conver in conversations if conver['from'] == 'gpt']), 'The number of human and gpt conversations are not equal'
        # cp the image to /Users/ssun2/Documents/VLM/HemaVisionQA-dev/HemeData/image_folder with the name of train or test + _index.jpeg
        # ignore if the image already exists
        if not os.path.exists('/Users/ssun2/Documents/VLM/HemaVisionQA-dev/HemeData/image_folder/'+str(_index)+'.jpeg'):
            shutil.copy('../Data/ucsf_data/data_preselected/'+diagnosis+'/'+image, '/Users/ssun2/Documents/VLM/HemaVisionQA-dev/HemeData/image_folder/'+ ["train","test"][test] + str(_index)+'.jpeg')
        id = ['train','test'][test] + str(_index)
        image = ['train','test'][test] + str(_index)+'.jpeg'
        json_files.append({'id':id, 
                           'image':image,
                           'conversations': conversations,
                           'diagnosis': diagnosis})
        # Open the JSON file in write mode  
    if os.path.exists(_dir):
        print('The file already exists')
        # overwrite the file
        print('Overwriting the file')
        os.remove(_dir)

    
    # Serializing json
    json_object = json.dumps(json_files, indent=4)

    with open(_dir, "w") as json_file:  
        # Iterate through the list of dictionaries  
         
        # Use `json.dump()` to write the dictionary to the file and add a newline character  
        json_file.write(json_object)
    return json_files
            





if __name__ == '__main__':

    # Import necessary modules  
    import pandas as pd  
    from sklearn.model_selection import train_test_split  
  
    ### create the dataframe
    df = create_df()
    ### create the json files

    # save dataframe
    df.to_csv('../Data/ucsf_data/LLaVA_heme.csv', index=False, sep = '\t')

    # Split the DataFrame into training and test sets with stratification  
    X_train, X_test, _,_ = train_test_split(  
        df,  # Features (X)  
        df["diagnosis"],               # Target (y)  
        test_size=0.2,                 # Test set size (20%)  
        stratify=df["diagnosis"],      # Stratification column  
        random_state=42                # For reproducibility  
    )  

    ### split the data by training and test, 0.8 and 0.2
    ### group by the diagnosis
    df_train = X_train
    df_test = X_test

    

    # data balance based on diagnosis
    # we want to make sure the data is balanced based on the diagnosis
    # and we balance them by upsampling

    # first we want to get the number of each diagnosis
    diagnosis = df_train['diagnosis'].unique()
    diagnosis_count = df_train['diagnosis'].value_counts()
    
    # we want to upsample the data based on the diagnosis
    spec_number = 2808 #max(diagnosis_count) 8000 kis too much we think 
    for diag in diagnosis:
        if diagnosis_count[diag] != spec_number:
            df_train_diag = df_train[df_train['diagnosis'] == diag]
            df_train_diag_upsample = df_train_diag.sample(spec_number, replace=True)
            # delete the original subset
            df_train = df_train.drop(df_train[df_train['diagnosis'] == diag].index)
            # add the upsampled subset
            df_train = pd.concat([df_train, df_train_diag_upsample], axis=0)
    # shuffle the data
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    print(df_train.shape)
    
    
    


    
    create_json_files(df_train, '/Users/ssun2/Documents/VLM/HemaVisionQA-dev/HemeData/LLaVA_heme_train.json')

    # generate a smaple of train.json of 30 data points
    df_train_sample = df_train.sample(30, replace=True)
    create_json_files(df_train_sample, '/Users/ssun2/Documents/VLM/HemaVisionQA-dev/HemeData/LLaVA_heme_train_sample.json')

    
    create_json_files(df_test, '/Users/ssun2/Documents/VLM/HemaVisionQA-dev/HemeData/LLaVA_heme_test.json', test = True)



    




    

    ### create the question and answers json file for test set
    # operating on the test data
    with open('/Users/ssun2/Documents/VLM/HemaVisionQA-dev/HemeData/LLaVA_heme_test.json') as f:
        data = json.load(f)

    
    # create a new list to store the data
    new_data_q = []
    new_data_a = []
    new_data_q_rl = []
    new_data_a_rl = []
    for i in range(len(data)):
        id = data[i]['id']
        image = data[i]['image']
        conversations = data[i]['conversations']
        diagnosis = data[i]['diagnosis']

        new_question = ''
        new_answer = ''
        

        for j in range(int(len(conversations)/2)):

            new_data_q.append({'question_id': id, 'image': image, 'text': conversations[2*j]['value'].replace('<image>','').replace('\n',''), 'category': diagnosis})

            new_data_a.append({'answer_id': id, 'image': image, 'text': conversations[2*j+1]['value'], 'category': diagnosis})

            new_question = new_question + conversations[2*j]['value'].replace('<image>','').replace('\n','') + ' '
            new_answer = new_answer + conversations[2*j+1]['value'] + ' '
        new_data_q_rl.append({'question_id': id, 'image': image, 'text': new_question, 'category': diagnosis})
        new_data_a_rl.append({'answer_id': id, 'image': image, 'text': new_answer, 'category': diagnosis})
        

        
    # save the new data
    # Open the JSON file in write mode
    if os.path.exists('/Users/ssun2/Documents/VLM/HemaVisionQA-dev/HemeData/LLaVA_heme_test_q.json'):
        print('The file already exists')
        # overwrite the file
        print('Overwriting the file')
        os.remove('/Users/ssun2/Documents/VLM/HemaVisionQA-dev/HemeData/LLaVA_heme_test_q.json')
    # Serializing json, use /n after each element in the list
    # data format 
    # {"question_id": 0, "image": "000000441147.jpg", "text": "What is the color of the two suitcases in the image?", "category": "AML"}
    # {"question_id": 1, "image": "000000441147.jpg", "text": "What is the color of the two suitcases in the image?", "category": "AML"}
    # for each row, we just have one question_id one image,one text and one category
    
    
    with open('/Users/ssun2/Documents/VLM/HemaVisionQA-dev/HemeData/LLaVA_heme_test_q.json', "w") as json_file:
        # Iterate through the list of dictionaries

        for element in new_data_q:  
        # Write each element to the file in JSON format, followed by a newline character  
            json_file.write(json.dumps(element) + "\n") 

    

    if os.path.exists('/Users/ssun2/Documents/VLM/HemaVisionQA-dev/HemeData/LLaVA_heme_test_a.json'):
        print('The file already exists')
        # overwrite the file
        print('Overwriting the file')
        os.remove('/Users/ssun2/Documents/VLM/HemaVisionQA-dev/HemeData/LLaVA_heme_test_a.json')
    # Serializing json, use /n after each json
    with open('/Users/ssun2/Documents/VLM/HemaVisionQA-dev/HemeData/LLaVA_heme_test_a.json', "w") as json_file:
        # Iterate through the list of dictionaries

        for element in new_data_a:  
        # Write each element to the file in JSON format, followed by a newline character  
            json_file.write(json.dumps(element) + "\n")

    # remove the test data becuase it is not useful anymore
    print('Removing the test data from the folder')
    os.remove('/Users/ssun2/Documents/VLM/HemaVisionQA-dev/HemeData/LLaVA_heme_test.json')





        
    


