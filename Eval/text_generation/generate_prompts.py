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
import random
import argparse
import shutil
import time
import glob
from tqdm import tqdm



sys.path.append('/wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RLHF/Eval/text_generation')

from text_repo import text_augmentation_repo

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


    
## { q_0,

            
def random_a_answer_sample(category): 
    category = category.replace('adequate_','').replace('_Mono','')
    category = category.upper() 
    categories =['NORMAL', 'AML', 'MM', 'CLOT', 'BLOOD']
    assert category in categories, f'the category is supposed to be in the ragne but it is not: {category}'
    answer_lists = []
    question_answers_set = text_augmentation_repo()
    #(category)
    
    
    logics = [1,1,1,1,1]
    
    assert len(logics) == 5, 'The length of logics is not 5'

    if category == 'NORMAL':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_01'))
        else:
            answer_lists.append(question_answers_set._sample('a_02'))
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_11'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_12'), \
                question_answers_set._sample('a_13')][_sample])
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_24'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_21'), \
                question_answers_set._sample('a_22'), \
                    question_answers_set._sample('a_23')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_241'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_211'), \
                question_answers_set._sample('a_221'), \
                    question_answers_set._sample('a_231')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2411'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2111'), \
                question_answers_set._sample('a_2211'), \
                    question_answers_set._sample('a_2311')][_sample])

    elif category == 'AML':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_01'))
        else:
            answer_lists.append(question_answers_set._sample('a_02'))
        
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_11'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_12'), \
                question_answers_set._sample('a_13')][_sample])
        
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_21'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_23'), \
                question_answers_set._sample('a_24')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_211'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_221'), \
                question_answers_set._sample('a_231'), \
                    question_answers_set._sample('a_241')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2111'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2211'), \
                question_answers_set._sample('a_2311'), \
                    question_answers_set._sample('a_2411')][_sample])
    
    elif category == 'MM':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_01'))
        else:
            answer_lists.append(question_answers_set._sample('a_02'))
        
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_11'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_12'), \
                question_answers_set._sample('a_13')][_sample])
        
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_21'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_23'), \
                question_answers_set._sample('a_24')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_221'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_211'), \
                question_answers_set._sample('a_231'), \
                    question_answers_set._sample('a_241')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2211'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2111'), \
                question_answers_set._sample('a_2311'), \
                    question_answers_set._sample('a_2411')][_sample])
    
    elif category == 'CLOT':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_02'))
        else:
            answer_lists.append(question_answers_set._sample('a_01'))
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_13'))
        else:
            
            answer_lists.append(
                question_answers_set._sample('a_11'))
            
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_23'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_21'), \
                question_answers_set._sample('a_24')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_231'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_211'), \
                question_answers_set._sample('a_221'), \
                    question_answers_set._sample('a_241')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2311'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2111'), \
                question_answers_set._sample('a_2211'), \
                    question_answers_set._sample('a_2411')][_sample])

    elif category == 'BLOOD':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_02'))
        else:
            answer_lists.append(question_answers_set._sample('a_01'))
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_12'))
        else: 
            answer_lists.append(
                question_answers_set._sample('a_11'))
            
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_23'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_21'), \
                question_answers_set._sample('a_24')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_231'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_211'), \
                question_answers_set._sample('a_221'), \
                    question_answers_set._sample('a_241')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2311'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2111'), \
                question_answers_set._sample('a_2211'), \
                    question_answers_set._sample('a_2411')][_sample])
        
        
        
    assert len(answer_lists) == 5, 'The length of answer_lists is not 5'
    return answer_lists



def random_conterfactual_prior(category, p): 

    category = category.replace('adequate_','').replace('_Mono','')
    category = category.upper() 
    categories =['NORMAL', 'AML', 'MM', 'CLOT', 'BLOOD']
    assert category in categories, f'the category is supposed to be in the ragne but it is not: {category}'
    assert p in [1,2,3,4], f'the actual value of p is {p}, which indicate it is not in the normal range'
    answer_lists = []
    question_answers_set = text_augmentation_repo()
    #(category)
    
    
    logics = [1,1,1,1,1]
    logics[p-1] = 0
    
    assert len(logics) == 5, 'The length of logics is not 5'
    

    if category == 'NORMAL':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_01'))
        else:
            answer_lists.append(question_answers_set._sample('a_02'))
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_11'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_12'), \
                question_answers_set._sample('a_13')][_sample])
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_24'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_21'), \
                question_answers_set._sample('a_22'), \
                    question_answers_set._sample('a_23')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_241'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_211'), \
                question_answers_set._sample('a_221'), \
                    question_answers_set._sample('a_231')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2411'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2111'), \
                question_answers_set._sample('a_2211'), \
                    question_answers_set._sample('a_2311')][_sample])
        

    elif category == 'AML':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_01'))
        else:
            answer_lists.append(question_answers_set._sample('a_02'))
        
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_11'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_12'), \
                question_answers_set._sample('a_13')][_sample])
        
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_21'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_23'), \
                question_answers_set._sample('a_24')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_211'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_221'), \
                question_answers_set._sample('a_231'), \
                    question_answers_set._sample('a_241')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2111'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2211'), \
                question_answers_set._sample('a_2311'), \
                    question_answers_set._sample('a_2411')][_sample])
    
    elif category == 'MM':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_01'))
        else:
            answer_lists.append(question_answers_set._sample('a_02'))
        
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_11'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_12'), \
                question_answers_set._sample('a_13')][_sample])
        
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_21'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_23'), \
                question_answers_set._sample('a_24')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_221'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_211'), \
                question_answers_set._sample('a_231'), \
                    question_answers_set._sample('a_241')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2211'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2111'), \
                question_answers_set._sample('a_2311'), \
                    question_answers_set._sample('a_2411')][_sample])
    
    elif category == 'CLOT':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_02'))
        else:
            answer_lists.append(question_answers_set._sample('a_01'))
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_13'))
        else:
            
            answer_lists.append(
                question_answers_set._sample('a_11'))
            
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_23'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_21'), \
                question_answers_set._sample('a_24')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_231'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_211'), \
                question_answers_set._sample('a_221'), \
                    question_answers_set._sample('a_241')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2311'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2111'), \
                question_answers_set._sample('a_2211'), \
                    question_answers_set._sample('a_2411')][_sample])

    elif category == 'BLOOD':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_02'))
        else:
            answer_lists.append(question_answers_set._sample('a_01'))
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_12'))
        else: 
            answer_lists.append(
                question_answers_set._sample('a_11'))
            
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_23'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_21'), \
                question_answers_set._sample('a_24')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_231'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_211'), \
                question_answers_set._sample('a_221'), \
                    question_answers_set._sample('a_241')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2311'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2111'), \
                question_answers_set._sample('a_2211'), \
                    question_answers_set._sample('a_2411')][_sample])
        
        
        
    assert len(answer_lists) == 5, 'The length of answer_lists is not 5'
    return answer_lists[p-1]

def random_consistent_conclusion(category, p): 

    category = category.replace('adequate_','').replace('_Mono','')
    category = category.upper() 
    categories =['NORMAL', 'AML', 'MM', 'CLOT', 'BLOOD']
    assert category in categories, f'the category is supposed to be in the ragne but it is not: {category}'
    
    answer_lists = []
    question_answers_set = text_augmentation_repo()
    #(category)
    
    
    logics = [1,1,1,1,1]
    
    
    assert len(logics) == 5, 'The length of logics is not 5'
    

    if category == 'NORMAL':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_01'))
        else:
            answer_lists.append(question_answers_set._sample('a_02'))
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_11'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_12'), \
                question_answers_set._sample('a_13')][_sample])
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_24'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_21'), \
                question_answers_set._sample('a_22'), \
                    question_answers_set._sample('a_23')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_241'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_211'), \
                question_answers_set._sample('a_221'), \
                    question_answers_set._sample('a_231')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2411'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2111'), \
                question_answers_set._sample('a_2211'), \
                    question_answers_set._sample('a_2311')][_sample])
        

    elif category == 'AML':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_01'))
        else:
            answer_lists.append(question_answers_set._sample('a_02'))
        
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_11'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_12'), \
                question_answers_set._sample('a_13')][_sample])
        
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_21'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_23'), \
                question_answers_set._sample('a_24')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_211'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_221'), \
                question_answers_set._sample('a_231'), \
                    question_answers_set._sample('a_241')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2111'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2211'), \
                question_answers_set._sample('a_2311'), \
                    question_answers_set._sample('a_2411')][_sample])
    
    elif category == 'MM':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_01'))
        else:
            answer_lists.append(question_answers_set._sample('a_02'))
        
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_11'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_12'), \
                question_answers_set._sample('a_13')][_sample])
        
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_21'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_23'), \
                question_answers_set._sample('a_24')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_221'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_211'), \
                question_answers_set._sample('a_231'), \
                    question_answers_set._sample('a_241')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2211'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2111'), \
                question_answers_set._sample('a_2311'), \
                    question_answers_set._sample('a_2411')][_sample])
    
    elif category == 'CLOT':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_02'))
        else:
            answer_lists.append(question_answers_set._sample('a_01'))
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_13'))
        else:
            
            answer_lists.append(
                question_answers_set._sample('a_11'))
            
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_23'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_21'), \
                question_answers_set._sample('a_24')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_231'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_211'), \
                question_answers_set._sample('a_221'), \
                    question_answers_set._sample('a_241')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2311'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2111'), \
                question_answers_set._sample('a_2211'), \
                    question_answers_set._sample('a_2411')][_sample])

    elif category == 'BLOOD':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_02'))
        else:
            answer_lists.append(question_answers_set._sample('a_01'))
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_12'))
        else: 
            answer_lists.append(
                question_answers_set._sample('a_11'))
            
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_23'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_21'), \
                question_answers_set._sample('a_24')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_231'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_211'), \
                question_answers_set._sample('a_221'), \
                    question_answers_set._sample('a_241')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2311'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2111'), \
                question_answers_set._sample('a_2211'), \
                    question_answers_set._sample('a_2411')][_sample])
        
        
        
    assert len(answer_lists) == 5, 'The length of answer_lists is not 5'
    return answer_lists[p]



def random_counterfactual_conclusion(category, p): 

    category = category.replace('adequate_','').replace('_Mono','')
    category = category.upper() 
    categories =['NORMAL', 'AML', 'MM', 'CLOT', 'BLOOD']
    assert category in categories, f'the category is supposed to be in the ragne but it is not: {category}'
    #assert p in [1,2,3,4], f'the actual value of p is {p}, which indicate it is not in the normal range'
    answer_lists = []
    question_answers_set = text_augmentation_repo()
    #(category)
    
    
    logics = [1,1,1,1,1]
    logics[p] =0
    
    
    assert len(logics) == 5, 'The length of logics is not 5'
    

    if category == 'NORMAL':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_01'))
        else:
            answer_lists.append(question_answers_set._sample('a_02'))
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_11'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_12'), \
                question_answers_set._sample('a_13')][_sample])
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_24'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_21'), \
                question_answers_set._sample('a_22'), \
                    question_answers_set._sample('a_23')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_241'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_211'), \
                question_answers_set._sample('a_221'), \
                    question_answers_set._sample('a_231')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2411'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2111'), \
                question_answers_set._sample('a_2211'), \
                    question_answers_set._sample('a_2311')][_sample])
        

    elif category == 'AML':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_01'))
        else:
            answer_lists.append(question_answers_set._sample('a_02'))
        
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_11'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_12'), \
                question_answers_set._sample('a_13')][_sample])
        
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_21'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_23'), \
                question_answers_set._sample('a_24')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_211'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_221'), \
                question_answers_set._sample('a_231'), \
                    question_answers_set._sample('a_241')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2111'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2211'), \
                question_answers_set._sample('a_2311'), \
                    question_answers_set._sample('a_2411')][_sample])
    
    elif category == 'MM':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_01'))
        else:
            answer_lists.append(question_answers_set._sample('a_02'))
        
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_11'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_12'), \
                question_answers_set._sample('a_13')][_sample])
        
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_21'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_23'), \
                question_answers_set._sample('a_24')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_221'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_211'), \
                question_answers_set._sample('a_231'), \
                    question_answers_set._sample('a_241')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2211'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2111'), \
                question_answers_set._sample('a_2311'), \
                    question_answers_set._sample('a_2411')][_sample])
    
    elif category == 'CLOT':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_02'))
        else:
            answer_lists.append(question_answers_set._sample('a_01'))
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_13'))
        else:
            
            answer_lists.append(
                question_answers_set._sample('a_11'))
            
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_23'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_21'), \
                question_answers_set._sample('a_24')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_231'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_211'), \
                question_answers_set._sample('a_221'), \
                    question_answers_set._sample('a_241')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2311'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2111'), \
                question_answers_set._sample('a_2211'), \
                    question_answers_set._sample('a_2411')][_sample])

    elif category == 'BLOOD':
        if logics[0] == 1:
            answer_lists.append(question_answers_set._sample('a_02'))
        else:
            answer_lists.append(question_answers_set._sample('a_01'))
        if logics[1] == 1:
            answer_lists.append(question_answers_set._sample('a_12'))
        else: 
            answer_lists.append(
                question_answers_set._sample('a_11'))
            
        if logics[2] == 1:
            answer_lists.append(question_answers_set._sample('a_23'))
        else:
            _sample = random.sample([0,1], 1)[0]
            answer_lists.append([question_answers_set._sample('a_21'), \
                question_answers_set._sample('a_24')][_sample])
        if logics[3] == 1:
            answer_lists.append(question_answers_set._sample('a_231'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_211'), \
                question_answers_set._sample('a_221'), \
                    question_answers_set._sample('a_241')][_sample])
        if logics[4] == 1:
            answer_lists.append(question_answers_set._sample('a_2311'))
        else:
            _sample = random.sample([0,1,2], 1)[0]
            answer_lists.append([question_answers_set._sample('a_2111'), \
                question_answers_set._sample('a_2211'), \
                    question_answers_set._sample('a_2411')][_sample])
        
        
        
    assert len(answer_lists) == 5, 'something is off'
    return answer_lists[p]

if __name__ == '__main__':

    pass