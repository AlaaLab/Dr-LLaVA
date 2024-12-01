import glob
import os
import sys
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random 
# convert a string of list to a list of strings, remove the order number
# example 


    
class  text_augmentation_repo():
    def __init__(self):

        ### create a text repo that is augumented from the llama chatbot
        self.train_repo = {}
        self.test_repo = {}
        self.create_text_repo_test()
        self.create_text_repo_train()

        self.soa =''#'<PAD>'
        self.eoa =''#'<PAD>'
        self.split = "train"
    def switch_split(self):
       if self.split == 'train':
           self.split = 'test'
       else:
           self.split = 'train'
    def _sample(self, key):
       # assert train_repo and test_repo should have the extact same keys
       # assert the key is in the train_repo and test_repo
       # assert the split is in ['train', 'test']
       # assert the key is in the train_repo and test_repo
       train_keys = list(self.train_repo.keys())
       test_keys = list(self.test_repo.keys())
       assert train_keys == test_keys, f'train and test keys are not the same, train_keys = {train_keys}, test_keys = {test_keys}'


       ### set the seed
       random.seed(0)
       if key is None:
           assert False, 'key is None'
       
       elif key not in self.test_repo.keys():
           # ramdon sample from 0 1
           assert False, 'key is not in the text repo'
       else:
           assert self.split in ['train', 'test'], 'split is not in [train, test]'
           if self.split == 'train':
               return np.random.choice(self.train_repo[key])    
           else:
               return np.random.choice(self.test_repo[key])     
    def create_text_repo_test(self):
        
       self.test_repo['q_0'] = [
    "Could you please examine this area of the bone marrow aspirate and decide if it can be used for diagnosing malignancies?",
    "Kindly assess this bone marrow aspirate region to determine whether it is suitable for malignancy diagnosis?",
    "Please analyze this part of the bone marrow aspirate and see if it can be utilized to diagnose malignant conditions?",
    "Would you evaluate this bone marrow aspirate section to determine if it can aid in diagnosing cancers?",
    "Could you inspect this region of the bone marrow aspirate and assess its suitability for malignancy diagnosis?"
  ]

       self.test_repo['a_01'] = ["The supplied pathology image is fit for assessing cancer presence.",
    "This pathology patch is appropriate for examining the presence of malignancy.",
    "The provided pathological sample is suitable for detecting cancer.",
    "This pathology image is adequate for assessing the existence of cancer.",
    "The pathology patch you provided is suitable for cancer analysis."]
       self.test_repo['a_02'] =    [ "This pathology sample cannot be used to make a diagnosis.",
    "The provided pathological patch is inadequate for diagnostic assessment.",
    "This pathological image is not acceptable for diagnosis.",
    "The given pathology image cannot be used for diagnostic evaluation.",
    "This pathology segment is unsuitable for diagnosis."]
       self.test_repo['a_03'] = ["This pathology sample cannot be used to make a diagnosis.",
    "The provided pathological patch is inadequate for diagnostic assessment.",
    "This pathological image is not acceptable for diagnosis.",
    "The given pathology image cannot be used for diagnostic evaluation.",
    "This pathology segment is unsuitable for diagnosis."]
       self.test_repo['q_1'] = [ "Could you describe the image's quality, focusing especially on any clotting, blood, or whether it is adequate for analysis?",
    "Please evaluate the quality of this image, particularly noting clotting, presence of blood, or its suitability for analysis?",
    "Kindly assess the image quality, focusing on whether it shows clotting, contains blood, or is adequate for analysis?",
    "Would you analyze the image's quality, especially noting any clotting, blood, or its adequacy for examination?",
    "Could you please examine the quality of the image, focusing on clotting, blood content, and its suitability for analysis?"]
       self.test_repo['a_11'] = [    "The high concentration of marrow nucleated cells makes this region ideal for diagnosis.",
    "This area is suitable for diagnosis due to abundant marrow nucleated cells.",
    "An ample presence of marrow nucleated cells renders this region excellent for diagnostic purposes.",
    "The region is optimal for diagnosis because of the high number of marrow nucleated cells.",
    "A significant abundance of marrow nucleated cells makes this area suitable for diagnostic evaluation."]
       self.test_repo['a_12'] = [  "An abundance of blood and red blood cells makes this region unsuitable for accurate diagnosis.",
    "The region is saturated with blood and numerous red blood cells, rendering it inadequate for precise evaluation.",
    "Due to excessive red blood cells, this blood-dominated area is not suitable for precise diagnosis.",
    "The high concentration of blood and red blood cells makes this region inappropriate for accurate medical diagnosis.",
    "This area shows an overwhelming presence of blood and red blood cells, unsuitable for accurate diagnosis."]
       self.test_repo['a_13'] = [  "Clotting with high particle density makes this region unsuitable for accurate diagnosis.",
    "The area contains a clot, which due to dense particles, is inadequate for precise medical evaluation.",
    "Because of clotting and dense particles, this region is inappropriate for accurate diagnosis.",
    "This region is unsuitable for precise medical diagnosis due to clotting characterized by dense particles.",
    "The presence of dense particles from clotting renders this area inadequate for accurate diagnosis."]
       self.test_repo['q_2'] = [    "Does the image reveal any irregularities that might indicate the presence of a disease?",
    "Are there any anomalies in the image that may suggest a disease?",
    "Can you observe any unusual features in the image that could indicate a disease is present?",
    "Does this image show any abnormalities that might suggest a disease is present?",
    "Are there any irregularities visible in this image that could indicate the presence of a disease?"]
       self.test_repo['a_21'] = [     "Yes, the image shows signs indicative of cancer.",
    "Indeed, the patch indicates patterns suggestive of malignancy.",
    "Affirmative, the image presents features consistent with a malignant condition.",
    "Certainly, the patch reveals evidence of cancer.",
    "Yes, the image demonstrates characteristics associated with malignancy."]
       self.test_repo['q_21'] = [
               "Have you seen any irregularities in cell growth within this patch of the image? If so, can you determine which cell type it is?",
    "Does the image patch display any abnormal cell increase? If yes, can you identify the specific cell type?",
    "Do you observe any unusual proliferation of cells in this area of the image? If so, can you specify the cell type?",
    "Are there any atypical growth patterns of cells in the image? If yes, can you name the cell type involved?",
    "Have you noticed any abnormal expansion of cells in the image patch? If so, can you determine the exact cell type?",
       ]
       self.test_repo['a_211'] = [     "An abnormal number of blast cells is present in the image.",
    "The patch shows an unusual proliferation of blast cells.",
    "Excessive blast cell growth is observed in this region.",
    "There is an abnormal accumulation of blast cells in the area.",
    "The image displays an overabundance of blast cells."]
       self.test_repo['q_211'] = [
    "What disease is indicated by this image patch?",
    "Which illness does the provided patch show?",
    "Can you identify the disease depicted in the given patch?",
    "What medical condition does this patch indicate?",
    "Which disease is suggested by the provided image patch?"
  ]
        
            
       self.test_repo['a_2111'] = [ 
    "The image suggests acute myeloid leukemia.",
    "Signs of acute myeloid leukemia are indicated by the patch.",
    "Features of acute myeloid leukemia are present in the image.",
    "The patch points toward acute myeloid leukemia.",
    "Evidence suggests possible acute myeloid leukemia."
  ]
        
       self.test_repo['a_22'] = [    "Yes, the image shows signs indicative of cancer.",
    "Indeed, the patch indicates patterns suggestive of malignancy.",
    "Affirmative, the image presents features consistent with a malignant condition.",
    "Certainly, the patch reveals evidence of cancer.",
    "Yes, the image demonstrates characteristics associated with malignancy."]
                                               
       self.test_repo['q_22'] = [    "Have you seen any irregularities in cell growth within this patch of the image? If so, can you determine which cell type it is?",
    "Does the image patch display any abnormal cell increase? If yes, can you identify the specific cell type?",
    "Do you observe any unusual proliferation of cells in this area of the image? If so, can you specify the cell type?",
    "Are there any atypical growth patterns of cells in the image? If yes, can you name the cell type involved?",
    "Have you noticed any abnormal expansion of cells in the image patch? If so, can you determine the exact cell type?"]
        
       self.test_repo['a_221'] = [     "An unusual proliferation of plasma cells is observed in the area.",
    "The image shows an abnormal increase in plasma cells.",
    "There is an excessive number of plasma cells in the patch.",
    "The region exhibits an overgrowth of plasma cells.",
    "An abnormal accumulation of plasma cells is present in the image."]
       self.test_repo['q_221'] = [      "What disease is indicated by this image patch?",
    "Which illness does the provided patch show?",
    "Can you identify the disease depicted in the given patch?",
    "What medical condition does this patch indicate?",
    "Which disease is suggested by the provided image patch?"]
       self.test_repo['a_2211'] = [      "The image indicates multiple myeloma.",
    "Signs of multiple myeloma are observed in the patch.",
    "Features consistent with multiple myeloma are present.",
    "The patch suggests the presence of multiple myeloma.",
    "Evidence points to possible multiple myeloma."]
        
        
       self.test_repo['a_23'] = [      "The image is too poor in quality to assess for irregularities.",
    "Due to insufficient quality, it's impossible to determine any irregularities.",
    "The low-quality image prevents evaluation of potential irregularities.",
    "Because the image is inadequate, assessing irregularities is not feasible.",
    "The poor image quality makes it impossible to detect any medical irregularities."]
       self.test_repo['q_23'] = [    "Have you seen any irregularities in cell growth within this patch of the image? If so, can you determine which cell type it is?",
    "Does the image patch display any abnormal cell increase? If yes, can you identify the specific cell type?",
    "Do you observe any unusual proliferation of cells in this area of the image? If so, can you specify the cell type?",
    "Are there any atypical growth patterns of cells in the image? If yes, can you name the cell type involved?",
    "Have you noticed any abnormal expansion of cells in the image patch? If so, can you determine the exact cell type?"]
       self.test_repo['a_231'] = [     "The sample lacks sufficient nucleated cells for analysis.",
    "Due to too few nucleated cells, the sample is inadequate.",
    "Analysis is not possible because the sample lacks nucleated cells.",
    "The insufficient nucleated cell count renders the sample unsuitable.",
    "An inadequate number of nucleated cells makes analysis impossible."]
       self.test_repo['q_231'] = [     "What disease is indicated by this image patch?",
    "Which illness does the provided patch show?",
    "Can you identify the disease depicted in the given patch?",
    "What medical condition does this patch indicate?",
    "Which disease is suggested by the provided image patch?"]
       self.test_repo['a_2311'] = [     "The image quality is too poor for an accurate diagnosis.",
    "An accurate diagnosis cannot be made due to low image quality.",
    "The inadequate quality of the image prevents precise diagnosis.",
    "Diagnosis is impossible because the image quality is insufficient.",
    "The poor image quality makes accurate diagnosis unfeasible."
  ]
        
       self.test_repo['a_24'] = [     "No, the image appears normal without observable irregularities.",
    "Negative, no irregularities are detected in the patch.",
    "No, the patch is normal with no signs of abnormalities.",
    "No, there are no detectable irregularities in the image.",
    "Negative, the image shows no signs of irregularities."]
        
       self.test_repo['q_24'] =  ["Have you seen any irregularities in cell growth within this patch of the image? If so, can you determine which cell type it is?",
    "Does the image patch display any abnormal cell increase? If yes, can you identify the specific cell type?",
    "Do you observe any unusual proliferation of cells in this area of the image? If so, can you specify the cell type?",
    "Are there any atypical growth patterns of cells in the image? If yes, can you name the cell type involved?",
    "Have you noticed any abnormal expansion of cells in the image patch? If so, can you determine the exact cell type?"]
       self.test_repo['a_241'] = [      "No, the image shows normal cellular growth without abnormalities.",
    "Negative, no unusual cell proliferation is observed.",
    "No abnormal cell growth is detected in the area.",
    "No, the patch displays typical cellular patterns.",
    "No signs of abnormal cell growth are present in the image."]
       self.test_repo['q_241'] = [     "What disease is indicated by this image patch?",
    "Which illness does the provided patch show?",
    "Can you identify the disease depicted in the given patch?",
    "What medical condition does this patch indicate?",
    "Which disease is suggested by the provided image patch?"]
       self.test_repo['a_2411'] = [      "No disease is indicated; the image appears normal.",
    "The patch shows no signs of disease and is normal.",
    "There is no evidence of disease; the image looks normal.",
    "The image appears normal with no disease indications.",
    "No signs of disease are present; the patch is normal."]
    
    def create_text_repo_train(self):
        
        self.train_repo['q_0'] = [
    "Could you please inspect this region of the bone marrow aspirate to determine if it is suitable for diagnosing malignancies?",
    "Kindly examine this area of the bone marrow aspirate and assess whether it can be utilized to diagnose malignant conditions?",
    "Please analyze this bone marrow aspirate section and decide if it can be used for malignancy diagnosis?",
    "Would you evaluate this region of the bone marrow aspirate to see if it is appropriate for diagnosing cancers?",
    "Could you examine this part of the bone marrow aspirate and determine if it is usable for malignancy diagnosis?",
    "Please assess this area of the bone marrow aspirate and determine whether it is suitable for diagnosing malignant diseases?",
    "Kindly inspect this bone marrow aspirate region and decide if it can be used to diagnose malignancies?",
    "Would you please examine this section of the bone marrow aspirate and assess its suitability for malignancy diagnosis?",
    "Could you analyze this area of the bone marrow aspirate to determine if it can aid in diagnosing malignancies?",
    "Please evaluate this part of the bone marrow aspirate and decide whether it can be used for malignancy diagnosis?",
    "Kindly assess this bone marrow aspirate area to determine if it is appropriate for diagnosing malignant conditions?",
    "Would you inspect this region of the bone marrow aspirate and see if it can be utilized to diagnose cancers?",
    "Could you please examine this section of the bone marrow aspirate and determine its suitability for malignancy diagnosis?",
    "Please analyze this area of the bone marrow aspirate and assess if it can be used to diagnose malignant diseases?",
    "Would you evaluate this bone marrow aspirate region to decide whether it is suitable for malignancy diagnosis?"
  ]

        self.train_repo['a_01'] = [    "The provided pathology patch is appropriate for examining the presence of cancer.",
    "This pathology image is suitable for assessing whether cancer is present.",
    "The given pathological sample can be used to detect cancer.",
    "This supplied pathology section is adequate for cancer examination.",
    "The pathology patch provided is suitable for evaluating the existence of cancer.",
    "This pathological image is acceptable for cancer assessment.",
    "The given pathology patch is appropriate for analyzing for cancer.",
    "The provided pathological sample is adequate for detecting cancer.",
    "This pathology image is fit for examining the presence of cancer.",
    "The pathology patch supplied is suitable for cancer diagnosis.",
    "This pathological sample is appropriate for assessing cancer presence.",
    "The provided pathology image is adequate for evaluating cancer.",
    "This given pathological patch can be used to examine for cancer.",
    "The pathology image provided is suitable for detecting cancer.",
    "This pathological section is acceptable for assessing the presence of cancer."]
        self.train_repo['a_02'] =    [ "The provided pathological image is not suitable for diagnosis.",
    "This pathology sample cannot be used for diagnostic purposes.",
    "The given pathology patch is inadequate for making a diagnosis.",
    "This pathological image is unsuitable for diagnostic use.",
    "The pathology segment provided is not fit for diagnosis.",
    "This pathology image is insufficient for diagnostic evaluation.",
    "The supplied pathological sample is not appropriate for diagnosis.",
    "This pathology patch is unsuitable for diagnostic assessment.",
    "The provided pathology image cannot be used for diagnosis.",
    "This pathological sample is inadequate for diagnostic purposes.",
    "The given pathological image is not suitable for diagnosis.",
    "This pathology segment is not appropriate for making a diagnosis.",
    "The pathological image provided is not fit for diagnostic use.",
    "This pathology patch is insufficient for diagnosis.",
    "The supplied pathological image is unsuitable for diagnostic purposes."]
        self.train_repo['a_03'] = ["The provided pathological image is not suitable for diagnosis.",
    "This pathology sample cannot be used for diagnostic purposes.",
    "The given pathology patch is inadequate for making a diagnosis.",
    "This pathological image is unsuitable for diagnostic use.",
    "The pathology segment provided is not fit for diagnosis.",
    "This pathology image is insufficient for diagnostic evaluation.",
    "The supplied pathological sample is not appropriate for diagnosis.",
    "This pathology patch is unsuitable for diagnostic assessment.",
    "The provided pathology image cannot be used for diagnosis.",
    "This pathological sample is inadequate for diagnostic purposes.",
    "The given pathological image is not suitable for diagnosis.",
    "This pathology segment is not appropriate for making a diagnosis.",
    "The pathological image provided is not fit for diagnostic use.",
    "This pathology patch is insufficient for diagnosis.",
    "The supplied pathological image is unsuitable for diagnostic purposes."]
        self.train_repo['q_1'] = [  "Could you please describe the image quality, particularly noting any clotting, blood, or adequacy for analysis?",
    "Kindly assess the quality of the image, especially if it shows clotting, contains blood, or is suitable for analysis?",
    "Please evaluate the image's quality, focusing on whether it displays clotting, blood, or is adequate for examination?",
    "Would you describe the quality of this image, particularly if it shows clotting, blood, or is appropriate for analysis?",
    "Could you assess the image quality, especially noting any clotting, presence of blood, or its adequacy for analysis?",
    "Please examine the image and describe its quality, focusing on clotting, blood, and suitability for analysis?",
    "Kindly evaluate the quality of the image, especially whether it indicates clotting, blood, or is sufficient for analysis?",
    "Would you please describe the image quality, particularly noting signs of clotting, blood presence, or adequacy for analysis?",
    "Could you analyze the image's quality, focusing especially on clotting, blood, and whether it is suitable for analysis?",
    "Please assess the quality of this image, particularly whether it shows clotting, has blood, or is adequate for analysis?",
    "Kindly describe the image quality, especially focusing on clotting, blood content, or its suitability for analysis?",
    "Would you evaluate the image quality, specifically noting any clotting, presence of blood, or adequacy for analysis?",
    "Could you please assess the quality of the image, focusing on whether it shows clotting, contains blood, or is fit for analysis?",
    "Please describe the quality of this image, especially noting any clotting, blood, or if it's adequate for analytical purposes?",
    "Would you examine the image and describe its quality, particularly if it shows clotting, blood, or is suitable for analysis?"]
        self.train_repo['a_11'] = ["The region contains a high percentage of marrow nucleated cells, making it optimal for diagnosis.",
    "This area has an abundance of marrow nucleated cells, rendering it suitable for diagnostic purposes.",
    "A high concentration of marrow nucleated cells in this region makes it ideal for diagnosis.",
    "The presence of numerous marrow nucleated cells makes this area excellent for diagnostic evaluation.",
    "This region is optimal for diagnosis due to the high percentage of marrow nucleated cells.",
    "An ample amount of marrow nucleated cells in this area makes it suitable for diagnosis.",
    "The high proportion of marrow nucleated cells renders this region ideal for diagnostic purposes.",
    "This area is excellent for diagnosis because of the abundance of marrow nucleated cells.",
    "The presence of many marrow nucleated cells makes this region optimal for diagnostic evaluation.",
    "A significant percentage of marrow nucleated cells in this area makes it suitable for diagnosis.",
    "This region is ideal for diagnosis due to the high concentration of marrow nucleated cells.",
    "An abundance of marrow nucleated cells renders this area excellent for diagnostic purposes.",
    "The high percentage of marrow nucleated cells in this region makes it optimal for diagnosis.",
    "This area contains numerous marrow nucleated cells, making it suitable for diagnostic evaluation.",
    "The presence of abundant marrow nucleated cells makes this region ideal for diagnosis."]
        self.train_repo['a_12'] = [  "The region is dominated by blood, with an abundance of red blood cells, making it unsuitable for accurate medical diagnosis.",
    "This area is filled with blood and numerous red blood cells, rendering it inadequate for precise diagnosis.",
    "An excess of red blood cells dominates the region, making it unsuitable for accurate medical evaluation.",
    "Due to the high concentration of blood and red blood cells, this region is not suitable for precise diagnosis.",
    "The abundance of red blood cells in this blood-dominated area makes it inappropriate for accurate diagnosis.",
    "This region shows a predominance of blood with many red blood cells, which is unsuitable for precise medical diagnosis.",
    "The excessive presence of red blood cells and blood renders this area inadequate for accurate evaluation.",
    "An overwhelming amount of blood and red blood cells make this region unsuitable for precise diagnosis.",
    "This area is saturated with blood and red blood cells, making it inappropriate for accurate medical evaluation.",
    "The high volume of red blood cells in this region dominated by blood is unsuitable for accurate diagnosis.",
    "Due to an abundance of blood and red blood cells, this area is inadequate for precise medical diagnosis.",
    "The dominance of red blood cells and blood in this region makes it unsuitable for accurate evaluation.",
    "This region, filled with blood and numerous red blood cells, is not suitable for precise diagnosis.",
    "An excessive amount of red blood cells and blood makes this area inappropriate for accurate medical diagnosis.",
    "The prevalence of blood and red blood cells in this region renders it inadequate for precise evaluation."]
        self.train_repo['a_13'] = [    "The region contains a clot with a high density of particles, making it unsuitable for accurate medical diagnosis.",
    "This area shows clotting characterized by dense particle concentration, rendering it inadequate for precise diagnosis.",
    "Due to a significant clot with dense particles, the region is unsuitable for accurate medical evaluation.",
    "The presence of a clot with dense particles in this region makes it inappropriate for accurate diagnosis.",
    "This region is characterized by clotting, resulting in high particle density, which is unsuitable for precise medical diagnosis.",
    "Clotting with dense particles is present, making the area unsuitable for accurate diagnosis.",
    "The high concentration of dense particles due to clotting renders this region inadequate for precise medical evaluation.",
    "This area contains a clot, making it unsuitable for accurate diagnosis because of the dense particle concentration.",
    "The clot in this region, marked by dense particles, makes it inappropriate for accurate medical diagnosis.",
    "Due to clotting and high particle density, this region is unsuitable for precise diagnosis.",
    "This region exhibits clotting with dense particles, rendering it inadequate for accurate medical evaluation.",
    "The presence of a clot characterized by dense particles makes this area unsuitable for precise diagnosis.",
    "Clotting in this region leads to high particle concentration, making it inappropriate for accurate diagnosis.",
    "This area shows a clot with dense particles, which is unsuitable for accurate medical diagnosis.",
    "The high density of particles due to clotting makes this region inadequate for precise medical evaluation."]
        self.train_repo['q_2'] = [ "Are there any abnormalities in the image that might indicate a disease?",
    "Does this image display any irregularities that could suggest the presence of a disease?",
    "Can you identify any anomalies in the image that may point to a disease?",
    "Does the image reveal any irregularities that might suggest a disease is present?",
    "Are there any unusual features in the image that could indicate the presence of a disease?",
    "Does this image show any signs that may suggest a disease is present?",
    "Can you observe any irregularities in the image that might indicate a disease?",
    "Are there any irregularities visible in the image that could suggest a disease?",
    "Does the image exhibit any anomalies that may indicate the presence of a disease?",
    "Can you detect any abnormalities in the image that might suggest a disease?",
    "Are there any signs in the image that could point to the presence of a disease?",
    "Does this image contain any irregularities that may suggest a disease?",
    "Can you find any unusual features in the image that could indicate a disease is present?",
    "Are there any irregularities in this image that might suggest the presence of a disease?",
    "Does the image show any anomalies that may indicate a disease is present?"]
        self.train_repo['a_21'] = [ "Yes, the patch shows patterns indicative of malignancy.",
    "Affirmative, the image exhibits signs suggestive of cancer.",
    "Indeed, the patch displays characteristics consistent with malignancy.",
    "Yes, the image reveals features indicative of a malignant condition.",
    "Certainly, the patch demonstrates patterns associated with cancer.",
    "Yes, the image shows evidence suggestive of malignancy.",
    "Indeed, the patch presents signs indicative of cancer.",
    "Yes, the image displays features consistent with a malignant condition.",
    "Affirmative, the patch reveals characteristics associated with cancer.",
    "Yes, the image indicates patterns suggestive of malignancy.",
    "Certainly, the patch exhibits signs consistent with cancer.",
    "Yes, the image demonstrates evidence indicative of a malignant condition.",
    "Indeed, the patch shows features associated with cancer.",
    "Yes, the image presents patterns suggestive of malignancy.",
    "Affirmative, the patch indicates characteristics consistent with cancer."]
        self.train_repo['q_21'] = ["Do you observe any abnormal cell proliferation in the image section? If yes, can you specify the cell type?",
    "Is there any atypical growth of cells in this image area? If so, could you identify the exact cell type?",
    "Have you detected any irregular cell growth within the image patch? If yes, can you determine the specific cell type?",
    "Are there signs of unusual cellular growth in the provided image patch? If so, can you name the cell type?",
    "Does the image section show any abnormal cell development? If yes, can you recognize the particular cell type?",
    "Do you see any uncommon cell expansion in this part of the image? If so, can you identify which cell type it is?",
    "Have you found any abnormal proliferation of cells in the image? If yes, can you specify the cell type involved?",
    "Is there any evidence of unusual cell multiplication in the image patch? If so, can you determine the exact cell type?",
    "Are any irregularities in cell growth present in the image section? If yes, can you name the specific cell type?",
    "Do you notice any abnormal cell formations in this image patch? If so, can you identify the cell type?",
    "Have you observed any atypical cell growth in the given image area? If yes, can you specify which cell type?",
    "Does the image show any irregular cellular proliferation? If so, can you determine the specific cell type?",
    "Is there any sign of unusual growth of cells in the image patch? If yes, can you identify the cell type?",
    "Do you detect any abnormal cellular activity in this image section? If so, can you name the exact cell type?",
    "Are there any signs of unusual cell division in the image? If yes, can you specify the particular cell type?"]
        self.train_repo['a_211'] = [ "An unusual proliferation of blast cells is observed in the provided region.",
    "There is an abnormal increase of blast cells in this image patch.",
    "The region shows an excessive growth of blast cells.",
    "An abnormal number of blast cells are present in the area.",
    "The image displays an unusual abundance of blast cells.",
    "An increased proliferation of blast cells is evident in this region.",
    "There is a notable overgrowth of blast cells in the patch.",
    "The area exhibits an unusual number of blast cells.",
    "An abnormal blast cell proliferation is observed in the image.",
    "The region shows excessive blast cell growth.",
    "An unusual increase in blast cells is present in this area.",
    "The image patch reveals an abnormal quantity of blast cells.",
    "An overabundance of blast cells is seen in the region.",
    "There is an unusual accumulation of blast cells in this patch.",
    "The area displays an abnormal proliferation of blast cells."]
        self.train_repo['q_211'] = [
    "Which disease is indicated by the provided image patch?",
    "What illness does the given patch suggest?",
    "Can you identify the disease shown in the provided patch?",
    "What condition does this image patch indicate?",
    "Which disease does the given patch represent?",
    "What ailment is suggested by the provided image patch?",
    "Can you tell what disease is depicted in the given patch?",
    "What disease does this patch show?",
    "Which illness is indicated by the image patch provided?",
    "What medical condition does the given patch suggest?",
    "Can you identify the condition shown in this image patch?",
    "What disease is represented by the provided patch?",
    "Which disease is shown in this image patch?",
    "What illness is indicated by the provided image patch?",
    "Can you determine the disease depicted in this patch?"
  ]
        
            
        self.train_repo['a_2111'] = [ 
    "The image patch suggests signs of acute myeloid leukemia.",
    "Findings indicate evidence of acute myeloid leukemia.",
    "The patch shows features consistent with acute myeloid leukemia.",
    "Signs of acute myeloid leukemia are present in the image.",
    "The image indicates possible acute myeloid leukemia.",
    "Features suggestive of acute myeloid leukemia are observed in the patch.",
    "The patch demonstrates characteristics of acute myeloid leukemia.",
    "Evidence of acute myeloid leukemia is present in the image.",
    "The image shows indications of acute myeloid leukemia.",
    "The patch reveals signs consistent with acute myeloid leukemia.",
    "Possible acute myeloid leukemia is suggested by the image.",
    "The findings are indicative of acute myeloid leukemia.",
    "Characteristics of acute myeloid leukemia are observed in the patch.",
    "The image displays features of acute myeloid leukemia.",
    "The patch suggests the presence of acute myeloid leukemia."
  ]
        
        self.train_repo['a_22'] = [    "Yes, the patch shows patterns indicative of malignancy.",
    "Affirmative, the image exhibits signs suggestive of cancer.",
    "Indeed, the patch displays characteristics consistent with malignancy.",
    "Yes, the image reveals features indicative of a malignant condition.",
    "Certainly, the patch demonstrates patterns associated with cancer.",
    "Yes, the image shows evidence suggestive of malignancy.",
    "Indeed, the patch presents signs indicative of cancer.",
    "Yes, the image displays features consistent with a malignant condition.",
    "Affirmative, the patch reveals characteristics associated with cancer.",
    "Yes, the image indicates patterns suggestive of malignancy.",
    "Certainly, the patch exhibits signs consistent with cancer.",
    "Yes, the image demonstrates evidence indicative of a malignant condition.",
    "Indeed, the patch shows features associated with cancer.",
    "Yes, the image presents patterns suggestive of malignancy.",
    "Affirmative, the patch indicates characteristics consistent with cancer."]
                                               
        self.train_repo['q_22'] = ["Do you observe any abnormal cell proliferation in the image section? If yes, can you specify the cell type?",
    "Is there any atypical growth of cells in this image area? If so, could you identify the exact cell type?",
    "Have you detected any irregular cell growth within the image patch? If yes, can you determine the specific cell type?",
    "Are there signs of unusual cellular growth in the provided image patch? If so, can you name the cell type?",
    "Does the image section show any abnormal cell development? If yes, can you recognize the particular cell type?",
    "Do you see any uncommon cell expansion in this part of the image? If so, can you identify which cell type it is?",
    "Have you found any abnormal proliferation of cells in the image? If yes, can you specify the cell type involved?",
    "Is there any evidence of unusual cell multiplication in the image patch? If so, can you determine the exact cell type?",
    "Are any irregularities in cell growth present in the image section? If yes, can you name the specific cell type?",
    "Do you notice any abnormal cell formations in this image patch? If so, can you identify the cell type?",
    "Have you observed any atypical cell growth in the given image area? If yes, can you specify which cell type?",
    "Does the image show any irregular cellular proliferation? If so, can you determine the specific cell type?",
    "Is there any sign of unusual growth of cells in the image patch? If yes, can you identify the cell type?",
    "Do you detect any abnormal cellular activity in this image section? If so, can you name the exact cell type?",
    "Are there any signs of unusual cell division in the image? If yes, can you specify the particular cell type?"]
        
        self.train_repo['a_221'] = [    "The region shows an unusual increase in plasma cells.",
    "An abnormal proliferation of plasma cells is observed in the image.",
    "There is an excessive number of plasma cells in this area.",
    "The image patch exhibits an unusual abundance of plasma cells.",
    "An increased growth of plasma cells is evident in the region.",
    "The area displays an abnormal increase in plasma cells.",
    "An unusual proliferation of plasma cells is present in this patch.",
    "There is a notable overgrowth of plasma cells in the image.",
    "The region reveals an abnormal number of plasma cells.",
    "An excessive proliferation of plasma cells is observed in this area.",
    "The image shows an unusual abundance of plasma cells.",
    "An overabundance of plasma cells is seen in the region.",
    "There is an abnormal accumulation of plasma cells in this patch.",
    "The area exhibits an unusual increase in plasma cells.",
    "An abnormal plasma cell proliferation is present in the image."]
        self.train_repo['q_221'] = [    "Which disease is indicated by the provided image patch?",
    "What illness does the given patch suggest?",
    "Can you identify the disease shown in the provided patch?",
    "What condition does this image patch indicate?",
    "Which disease does the given patch represent?",
    "What ailment is suggested by the provided image patch?",
    "Can you tell what disease is depicted in the given patch?",
    "What disease does this patch show?",
    "Which illness is indicated by the image patch provided?",
    "What medical condition does the given patch suggest?",
    "Can you identify the condition shown in this image patch?",
    "What disease is represented by the provided patch?",
    "Which disease is shown in this image patch?",
    "What illness is indicated by the provided image patch?",
    "Can you determine the disease depicted in this patch?"]
        self.train_repo['a_2211'] = [  "The findings indicate potential evidence of multiple myeloma.",
    "The image patch suggests signs of multiple myeloma.",
    "Features consistent with multiple myeloma are observed in the image.",
    "Signs of multiple myeloma are present in the patch.",
    "The image indicates possible multiple myeloma.",
    "The patch shows characteristics of multiple myeloma.",
    "Evidence of multiple myeloma is suggested by the findings.",
    "The image reveals features consistent with multiple myeloma.",
    "Possible multiple myeloma is indicated by the patch.",
    "The findings are indicative of multiple myeloma.",
    "Characteristics of multiple myeloma are observed in the image.",
    "The patch displays signs of multiple myeloma.",
    "The image suggests the presence of multiple myeloma.",
    "Features of multiple myeloma are evident in the patch.",
    "The findings point toward multiple myeloma."]
        
        
        self.train_repo['a_23'] = [  "The image patch is of insufficient quality, making it impossible to determine the presence of irregularities.",
    "Due to poor quality, the image cannot be evaluated for irregularities indicating a medical condition.",
    "The inadequate image quality prevents assessment of potential irregularities.",
    "Because the image is of low quality, it's impossible to detect any irregularities.",
    "The insufficient quality of the image patch makes it impossible to determine any medical irregularities.",
    "The image quality is too poor to assess for irregularities that could indicate a condition.",
    "Due to the inadequate image, determining the presence of irregularities is impossible.",
    "The low-quality image prevents detection of any possible irregularities.",
    "Because of insufficient image quality, assessing irregularities is not possible.",
    "The poor quality of the image makes it impossible to identify any irregularities.",
    "The image patch is too inadequate to determine the presence of medical irregularities.",
    "Due to the image's poor quality, it's impossible to evaluate for irregularities.",
    "The insufficient image quality prevents identification of any potential irregularities.",
    "Because the image is inadequate, detecting irregularities is impossible.",
    "The low quality of the image patch makes assessment of irregularities impossible."]
        self.train_repo['q_23'] = ["Do you observe any abnormal cell proliferation in the image section? If yes, can you specify the cell type?",
    "Is there any atypical growth of cells in this image area? If so, could you identify the exact cell type?",
    "Have you detected any irregular cell growth within the image patch? If yes, can you determine the specific cell type?",
    "Are there signs of unusual cellular growth in the provided image patch? If so, can you name the cell type?",
    "Does the image section show any abnormal cell development? If yes, can you recognize the particular cell type?",
    "Do you see any uncommon cell expansion in this part of the image? If so, can you identify which cell type it is?",
    "Have you found any abnormal proliferation of cells in the image? If yes, can you specify the cell type involved?",
    "Is there any evidence of unusual cell multiplication in the image patch? If so, can you determine the exact cell type?",
    "Are any irregularities in cell growth present in the image section? If yes, can you name the specific cell type?",
    "Do you notice any abnormal cell formations in this image patch? If so, can you identify the cell type?",
    "Have you observed any atypical cell growth in the given image area? If yes, can you specify which cell type?",
    "Does the image show any irregular cellular proliferation? If so, can you determine the specific cell type?",
    "Is there any sign of unusual growth of cells in the image patch? If yes, can you identify the cell type?",
    "Do you detect any abnormal cellular activity in this image section? If so, can you name the exact cell type?",
    "Are there any signs of unusual cell division in the image? If yes, can you specify the particular cell type?"]
        self.train_repo['a_231'] = [    "The sample is inadequate, with an insufficient number of nucleated cells for analysis.",
    "Due to a lack of nucleated cells, the sample is insufficient for evaluation.",
    "The sample cannot be analyzed because it lacks enough nucleated cells.",
    "An insufficient number of nucleated cells makes the sample inadequate for analysis.",
    "The sample is unsuitable for analysis due to too few nucleated cells.",
    "There are not enough nucleated cells in the sample for proper evaluation.",
    "The insufficient nucleated cell count renders the sample inadequate for analysis.",
    "Analysis is impossible due to the sample's lack of nucleated cells.",
    "The sample is inadequate for evaluation because of insufficient nucleated cells.",
    "An inadequate number of nucleated cells makes the sample unsuitable for analysis.",
    "The sample cannot be analyzed due to an insufficient nucleated cell count.",
    "The lack of enough nucleated cells renders the sample inadequate for evaluation.",
    "Due to too few nucleated cells, the sample is unsuitable for analysis.",
    "The sample is insufficient for analysis because it lacks nucleated cells.",
    "An insufficient nucleated cell number makes the sample inadequate for evaluation."]
        self.train_repo['q_231'] = [    "Which disease is indicated by the provided image patch?",
    "What illness does the given patch suggest?",
    "Can you identify the disease shown in the provided patch?",
    "What condition does this image patch indicate?",
    "Which disease does the given patch represent?",
    "What ailment is suggested by the provided image patch?",
    "Can you tell what disease is depicted in the given patch?",
    "What disease does this patch show?",
    "Which illness is indicated by the image patch provided?",
    "What medical condition does the given patch suggest?",
    "Can you identify the condition shown in this image patch?",
    "What disease is represented by the provided patch?",
    "Which disease is shown in this image patch?",
    "What illness is indicated by the provided image patch?",
    "Can you determine the disease depicted in this patch?"]
        self.train_repo['a_2311'] = [    "The image quality is poor, making it unsuitable for an accurate diagnosis.",
    "Due to low image quality, an accurate diagnosis is not possible.",
    "The poor quality of the image makes it unsuitable for precise diagnosis.",
    "An accurate diagnosis cannot be made because the image quality is inadequate.",
    "The image is of insufficient quality for an accurate diagnosis.",
    "Because of poor image quality, the diagnosis cannot be accurately made.",
    "The inadequate image quality renders it unsuitable for precise diagnosis.",
    "An accurate diagnosis is impossible due to the low quality of the image.",
    "The image quality is too poor for a precise diagnosis.",
    "Due to inadequate image quality, an accurate diagnosis cannot be made.",
    "The poor quality of the image prevents an accurate diagnosis.",
    "An accurate diagnosis is not possible because the image quality is insufficient.",
    "The image is unsuitable for precise diagnosis due to poor quality.",
    "Because the image quality is low, an accurate diagnosis cannot be achieved.",
    "The inadequate quality of the image makes it impossible to make an accurate diagnosis."
  ]
        
        self.train_repo['a_24'] = [    "There is no indication of disease; the patch appears normal.",
    "No signs of disease are present; the image looks normal.",
    "The patch appears normal with no evidence of disease.",
    "No indications of disease are observed; the image is normal.",
    "The image shows no signs of disease and appears normal.",
    "No evidence of disease is present; the patch looks normal.",
    "The patch is normal with no indications of disease.",
    "No signs of disease are observed; the image appears normal.",
    "The image appears normal with no evidence of disease.",
    "There are no indications of disease; the patch is normal.",
    "No signs of disease are present; the patch appears normal.",
    "The patch shows no evidence of disease and looks normal.",
    "No disease indications are observed; the image is normal.",
    "The image is normal with no signs of disease.",
    "No evidence of disease is seen; the patch appears normal."]
        
        self.train_repo['q_24'] =  ["Do you observe any abnormal cell proliferation in the image section? If yes, can you specify the cell type?",
    "Is there any atypical growth of cells in this image area? If so, could you identify the exact cell type?",
    "Have you detected any irregular cell growth within the image patch? If yes, can you determine the specific cell type?",
    "Are there signs of unusual cellular growth in the provided image patch? If so, can you name the cell type?",
    "Does the image section show any abnormal cell development? If yes, can you recognize the particular cell type?",
    "Do you see any uncommon cell expansion in this part of the image? If so, can you identify which cell type it is?",
    "Have you found any abnormal proliferation of cells in the image? If yes, can you specify the cell type involved?",
    "Is there any evidence of unusual cell multiplication in the image patch? If so, can you determine the exact cell type?",
    "Are any irregularities in cell growth present in the image section? If yes, can you name the specific cell type?",
    "Do you notice any abnormal cell formations in this image patch? If so, can you identify the cell type?",
    "Have you observed any atypical cell growth in the given image area? If yes, can you specify which cell type?",
    "Does the image show any irregular cellular proliferation? If so, can you determine the specific cell type?",
    "Is there any sign of unusual growth of cells in the image patch? If yes, can you identify the cell type?",
    "Do you detect any abnormal cellular activity in this image section? If so, can you name the exact cell type?",
    "Are there any signs of unusual cell division in the image? If yes, can you specify the particular cell type?"]
        self.train_repo['a_241'] = [    "No, the image patch does not display any abnormal cellular growth.",
    "Negative, there is no unusual cell proliferation observed in the region.",
    "No abnormal cell growth is present in this area.",
    "No, the patch shows normal cellular patterns without abnormalities.",
    "No unusual cell growth is detected in the image.",
    "Negative, the area displays normal cell development.",
    "No, the image shows no signs of abnormal cell proliferation.",
    "No abnormal cellular activity is observed in the patch.",
    "No, the region appears normal with typical cell growth.",
    "No unusual cellular growth is present in this image patch.",
    "Negative, the image displays normal cell patterns.",
    "No, there is no abnormal cell growth detected in the area.",
    "No signs of unusual cell proliferation are observed in the image.",
    "No, the patch shows normal cellular structures.",
    "No abnormal cell development is present in this region."]
        self.train_repo['q_241'] = [    "Which disease is indicated by the provided image patch?",
    "What illness does the given patch suggest?",
    "Can you identify the disease shown in the provided patch?",
    "What condition does this image patch indicate?",
    "Which disease does the given patch represent?",
    "What ailment is suggested by the provided image patch?",
    "Can you tell what disease is depicted in the given patch?",
    "What disease does this patch show?",
    "Which illness is indicated by the image patch provided?",
    "What medical condition does the given patch suggest?",
    "Can you identify the condition shown in this image patch?",
    "What disease is represented by the provided patch?",
    "Which disease is shown in this image patch?",
    "What illness is indicated by the provided image patch?",
    "Can you determine the disease depicted in this patch?"]
        self.train_repo['a_2411'] = [    "There is no indication of disease; the patch appears normal.",
    "No signs of disease are present; the image looks normal.",
    "The patch appears normal with no evidence of disease.",
    "No indications of disease are observed; the image is normal.",
    "The image shows no signs of disease and appears normal.",
    "No evidence of disease is present; the patch looks normal.",
    "The patch is normal with no indications of disease.",
    "No signs of disease are observed; the image appears normal.",
    "The image appears normal with no evidence of disease.",
    "There are no indications of disease; the patch is normal.",
    "No signs of disease are present; the patch appears normal.",
    "The patch shows no evidence of disease and looks normal.",
    "No disease indications are observed; the image is normal.",
    "The image is normal with no signs of disease.",
    "No evidence of disease is seen; the patch appears normal."]
        
def convert_string_to_list(string):
    lists = string.split('\n')
    ### remove the order number
    lists = [re.sub(r'^\d+\.\s+', '', i) for i in lists]
    return lists

def hint_style_answer(agreement, answer):
    ### input is a string, the output is a query with this template
    ### if agreement == True, then the answer just answer
    ### if agreement == False, then the answer is 'I don't agree with you. I think <answer>.'
    if agreement:
        return answer
    else:
        return 'I don\'t agree with you. I think '+answer
    
def hint_style_question(prior_answers, query):
    ### input is a string, the output is a query with this template
    ### Doctor believe <prior_answers>, <query>?

    return 'Doctor believe '+prior_answers+' '+query

def confirmation_style_question(prior_answers, query):
    ### input is a string, the output is a query with this template
    ### Doctor believe <prior_answers>, <query>?

    return 'Doctor believe '+prior_answers+' Do you agree with me?'

def confirmation_style_answer(agreement,  answer):
    ### input is a string, the output is a query with this template
    ### if agreement == True, then the answer just answer
    ### if agreement == False, then the answer is 'I don't agree with you. I think <answer>.'
    if agreement:
        return answer
    else:
        return 'I don\'t agree with you. I think '+answer

### mian function 
if __name__ == '__main__':
    assert 1==2, 'this script is not supposed to be run'
    
