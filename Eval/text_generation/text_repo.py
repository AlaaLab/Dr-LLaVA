import glob
import os
import sys
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# convert a string of list to a list of strings, remove the order number
# example


def convert_string_to_list(string):
    lists = string.split("\n")
    ### remove the order number
    lists = [re.sub(r"^\d+\.\s+", "", i) for i in lists]
    return lists


class text_augmentation_repo:
    def __init__(self):

        ### create a text repo that is augumented from the llama chatbot
        self.text_repo = {}
        self.create_text_repo()
        self.soa = ""  #'<PAD>'
        self.eoa = ""  #'<PAD>'

    def _sample(self, key):
        if key is None:
            assert False, "key is None"
        elif key not in self.text_repo.keys():
            assert False, "key is not in the text repo"
        else:
            if "a_" in key:
                # key is an answer
                return self.soa + np.random.choice(self.text_repo[key]) + "." + self.eoa
            else:
                # key is a question or observation
                return np.random.choice(self.text_repo[key])

    def create_text_repo(self):
        self.text_repo["Observation"] = [
            "The prior assessment you made is",
            "The earlier evaluation is as noted",
            "The prior assessment you made is",
            "Your earlier examination stands as",
            "The earlier assessment you made is",
            "The prior examination you conducted is",
            "The prior assessment you made is",
            "The earlier examination conducted is",
            "The prior assessment you made is",
            "The earlier examination you conducted is",
        ]

        self.text_repo["q_0"] = [
            "Examining these pathology images, can you determine if they can be utilized for cancer diagnosis?",
            "Can you assess if these pathology images are suitable for identifying cancer upon inspection?",
            "By analyzing these pathology images, can you confirm if they are appropriate for cancer detection?",
            "Upon reviewing these pathology images, would you say they can be employed for diagnosing cancer?",
            "Can you evaluate these pathology images and ascertain if they're useful for cancer diagnosis purposes?",
            "Examining these pathology pictures, can you determine if they are suitable for cancer diagnosis?",
            "Can you assess whether these pathology images can be utilized for diagnosing cancer?",
            "Upon reviewing these pathological images, can you confirm if they are appropriate for cancer detection?",
            "Analyzing these pathology visuals, can you ascertain if they can be employed in identifying cancer?",
            "Considering these pathological photographs, can you verify if they can be used for cancer diagnosis purposes?",
        ]
        self.text_repo["a_01"] = [
            "This pathology image section can be effectively utilized for diagnostic purposes in the medical field",
            "The provided pathology image fragment is appropriate for determining diagnoses, including cancer identification through analysis",
            "This pathology image segment is appropriate for use in making a medical diagnosis",
            "This pathological picture piece can effectively be utilized for diagnostic purposes in identifying cancer",
            "This pathology image segment can be effectively utilized for diagnostic purposes in the medical field",
            "The provided pathology visual section is appropriate for determining diagnoses, potentially including cancer identification",
            "This pathology image section can effectively be utilized in making a medical diagnosis",
            "The given pathology picture fragment is appropriate for determining a diagnosis in the context of cancer identification",
            "This pathology image segment is appropriate for conducting medical diagnoses",
            "This pathological picture section is apt for determining diagnoses in the medical field",
        ]
        self.text_repo["a_02"] = [
            "This pathological image segment cannot be utilized for accurate medical diagnosis purposes",
            "For cancer diagnosis purposes, these pathological photographs are not appropriate to use for conclusive determinations",
            "This pathological image segment cannot be utilized for accurate medical diagnosis",
            "The provided pathology image piece is inadequate for determining a cancer diagnosis",
            "For cancer diagnosis purposes, this particular pathology image section is unsuitable",
            "The given pathology image patch cannot be utilized for accurate medical diagnosis purposes",
            "For cancer diagnosis purposes, these pathological photographs are not appropriate to be used",
            "These pathology image patches are unsuitable for providing accurate medical diagnosis, particularly for cancer detection",
            "The given pathological image patch cannot be reliably utilized for cancer diagnosis purposes",
        ]
        self.text_repo["a_03"] = [
            "This pathological image segment cannot be utilized for accurate medical diagnosis purposes",
            "For cancer diagnosis purposes, these pathological photographs are not appropriate to use for conclusive determinations",
            "This pathological image segment cannot be utilized for accurate medical diagnosis",
            "The provided pathology image piece is inadequate for determining a cancer diagnosis",
            "For cancer diagnosis purposes, this particular pathology image section is unsuitable",
            "The given pathology image patch cannot be utilized for accurate medical diagnosis purposes",
            "For cancer diagnosis purposes, these pathological photographs are not appropriate to be used",
            "These pathology image patches are unsuitable for providing accurate medical diagnosis, particularly for cancer detection",
            "The given pathological image patch cannot be reliably utilized for cancer diagnosis purposes",
        ]
        self.text_repo["q_1"] = [
            "Could you provide a description of the pathology image patch's quality?",
            "Could you please provide details on the clarity of the pathology image section?",
            "Could you provide details on the clarity of the pathology image section?",
            "Could you elaborate on the clarity and details of the pathological image segment?",
            "Could you provide details on the clarity of the pathology image section?",
            "Could you provide a description of the quality of the pathology image segment?",
            "Could you provide a description of the pathology image patch's quality?",
            "Could you provide details on the clarity of the pathology image section?",
            "Could you provide details about the quality of the pathology image segment?",
            "Could you provide details on the quality of the pathology image section?",
        ]
        self.text_repo["a_11"] = [
            "The area is in proximity to aspirated particles containing a high percentage of marrow nucleated cells, making it optimal for diagnostic objectives",
            "In the vicinity of aspirate particles, the presence of a large proportion of marrow nucleated cells is advantageous for diagnostic purposes",
            "The area is close to aspirated particles containing a high percentage of marrow nucleated cells, making it suitable for diagnostic objectives",
            "The vicinity is adjacent to particles with aspirated characteristics and a high concentration of marrow nucleated cells, which is optimal for diagnostic applications",
            "The area is in close proximity to aspirate particles containing a significant number of marrow nucleated cells, making it optimal for diagnostic purposes",
            "In the vicinity of aspirate particles, the presence of high percentages of marrow nucleated cells makes this region well-suited for diagnostic evaluations",
            "In the vicinity of aspirate particles, there is a significant concentration of marrow nucleated cells, making it a prime location for diagnostic examination",
            "The area adjacent to aspirate particles contains a large percentage of marrow nucleated cells, creating an optimal environment for diagnostic evaluations",
            "The area is proximate to aspirated particles containing a significant percentage of marrow nucleated cells, making it optimal for diagnostic applications",
            "The vicinity is adjacent to particles with a high proportion of marrow nucleated cells, which is highly suitable for diagnostic objectives",
        ]
        self.text_repo["a_12"] = [
            "The area exhibits a high concentration of red blood cells, making it unsuitable for accurate medical diagnosis",
            "With an elevated presence of RBCs, the section is blood-rich and not optimal for diagnostic purposes",
            "The area exhibits a substantial presence of red blood cells, making it unsuitable for accurate medical diagnosis",
            "With an elevated concentration of RBCs, the bloody section is not optimal for diagnostic purposes in pathology",
            "The area exhibits an abundance of red blood cells, making it unsuitable for accurate medical diagnosis",
            "With a high concentration of RBCs present, this blood-rich area is not conducive to precise diagnostic evaluation",
            "The area exhibits a significant presence of red blood cells, making it less suitable for accurate medical diagnosis",
            "High concentrations of RBCs are present in the bloody area, which hinders the effectiveness of diagnostic procedures",
            "The area exhibits a high concentration of red blood cells, creating a bloody appearance that is not optimal for diagnostic purposes",
            "With an abundance of RBCs present, the region appears bloody, making it less than ideal for accurate medical diagnosis",
        ]
        self.text_repo["a_13"] = [
            "The area displays a high concentration of dense aspirate particles, making it unsuitable for accurate medical diagnosis",
            "In this zone, there is a significant presence of thick aspirate particles, which hinders precise diagnostic efforts",
            "In this area, there is a high concentration of thick aspirate particles, which negatively impacts the diagnostic process",
            "The presence of substantial dense aspirate particles in this section hinders accurate medical diagnosis",
            "In this area, there is a high concentration of dense aspirate particles, making it unsuitable for diagnostic purposes",
            "The presence of numerous dense aspirate particles in this region hinders accurate medical diagnosis",
            "The area comprises a high concentration of thick aspirate particles, adversely affecting the diagnostic process",
            "In this section, there is a significant presence of dense aspirate particles, making it unsuitable for accurate diagnosis",
            "The area consists of a high concentration of thick aspirate particles, making it unsuitable for diagnostic purposes",
            "In this zone, there is a significant presence of dense aspirate particles, which negatively impacts the diagnostic process",
        ]
        self.text_repo["q_2"] = [
            "Can you identify any irregularities within the image patch that could potentially suggest the presence of a disease?",
            "Can you identify any irregularities within the image patch that could suggest the presence of a disease?",
            "Can you identify any irregularities in the image patch that could potentially suggest the presence of an illness?",
            "Can you identify any irregularities in the image patch that may suggest the presence of a medical condition?",
            "Can you identify any irregularities in the image patch that could potentially suggest a medical condition?",
            "Could you identify any irregularities within the image patch that may suggest the presence of a medical condition?",
            "Could you identify any irregularities within the image patch that may suggest the presence of a medical condition?",
            "Can you identify any irregularities within the image patch that could potentially suggest a medical condition?",
            "Could you identify any irregularities within the image patch that may suggest the presence of a disease?",
            "Could you identify any irregularities in the image patch that may suggest the presence of a medical condition?",
        ]
        self.text_repo["a_21"] = [
            "The patch exhibits patterns indicative of malignancy",
            "Malignant patterns are displayed in the patch",
            "The image patch displays patterns indicative of malignancy",
            "Malignant patterns are evident in the patch under examination",
            "The image patch displays malignant characteristics",
            "Malignancy patterns are evident in the patch",
            "The patch displays patterns indicative of malignancy",
            "Malignant patterns are evident in the patch",
            "The image patch displays patterns indicative of malignancy",
            "Malignant patterns are evident in the patch under examination",
        ]
        self.text_repo["q_21"] = [
            "Have you noticed any unusual cell growth within the image patch, and can you identify the specific cell type?",
            "Have you noticed any unusual cell growth in the image section, and can you identify the specific cell type?",
            "Have you noticed any unusual cell growth in the image section, and can you identify the specific cell type?",
            "Have you identified any unusual cell growth within the image patch, and if so, can you specify the cell type?",
            "Have you noticed any unusual cell growth within the image patch, and if so, can you identify the specific cell type?",
            "Have you identified any unusual cell growth in the image section, and can you specify the cell type involved?",
            "Have you noticed any unusual cell growth within the image patch, and if so, can you identify the specific cell type?",
            "Have you noticed any unusual cell growth within the image patch, and can you identify the specific cell type?",
            "Have you noticed any unusual growth of cells, and if so, can you identify the specific cell type?",
            "Have you noticed any unusual cellular growth within the image section, and can you identify the specific cell type involved?",
        ]
        self.text_repo["a_211"] = [
            "The picture segment exhibits an atypical increase in Myeloblast cell numbers",
            "An unusual expansion of Myeloblast cells is present within the visualized area",
            "The image segment displays an atypical increase in the number of Myeloblast cells",
            "In the image section, there is an irregular expansion of Myeloblast cells present",
            "The picture segment displays an atypical increase in the number of Myeloblast cells",
            "An unusual expansion of Myeloblast cells is observed in the image portion",
            "The picture segment exhibits an unusual increase in the number of Myeloblast cells",
            "An atypical multiplication of Myeloblast cells is evident in the image section",
            "The picture segment exhibits an atypical increase in Myeloblast cell population",
            "An irregular expansion of Myeloblast cells is observed in the visual snippet",
        ]
        self.text_repo["q_211"] = [
            "As a pathologist, would you be able to determine the possible blood cancer from the image patch provided?",
            "Can you pinpoint the probable illness depicted in this image patch, acting as a pathologist?",
            "Could you discern the potential condition represented in the image patch?",
            "Can you determine the possible illness depicted in this image section?",
            "Would you be able to recognize the probable disease present in the visual fragment as a pathologist?",
            "Can you pinpoint the potential disorder shown in the image piece?",
            "Can you determine the possible ailment depicted in this image patch?",
            "Can you pinpoint the likely condition shown in the image patch?",
            "Are you able to recognize the probable illness in the image patch?",
            "Can you determine the possible illness from the provided image patch?",
            "Are you able to recognize the likely condition depicted in this image patch?",
            "Would you be able to diagnose the probable ailment in the image patch?",
            "Can you determine the possible illness depicted in this image patch, as a pathologist would?",
            "As a pathologist, are you able to pinpoint the likely ailment present in the image section?",
            "Could you recognize the probable condition displayed in the image fragment?",
        ]

        self.text_repo["a_2111"] = [
            "In this image patch, there is evidence suggestive of acute myeloid leukemia, as a pathologist might identify",
            "The visual representation in the image patch indicates the potential presence of acute myeloid leukemia, similar to a pathologist's determination",
            "The picture segment displays a sign of acute myeloid leukemia",
            "In the provided image section, there is an evidence of acute myeloid leukemia",
            "The provided image patch reveals a sign of acute myeloid leukemia",
            "An indication of acute myeloid leukemia is displayed in the image patch",
            "The image patch displays signs suggestive of acute myeloid leukemia",
            "Acute myeloid leukemia is indicated by the image patch presented",
            "The image patch displays a sign of acute myeloid leukemia",
            "In the image patch, there is an indication of acute myeloid leukemia being present",
        ]

        self.text_repo["a_22"] = [
            "The patch exhibits patterns indicative of malignancy",
            "Malignant patterns are displayed in the patch",
            "The image patch displays patterns indicative of malignancy",
            "Malignant patterns are evident in the patch under examination",
            "The image patch displays malignant characteristics",
            "Malignancy patterns are evident in the patch",
            "The patch displays patterns indicative of malignancy",
            "Malignant patterns are evident in the patch",
            "The image patch displays patterns indicative of malignancy",
            "Malignant patterns are evident in the patch under examination",
        ]

        self.text_repo["q_22"] = [
            "Have you noticed any unusual cell growth within the image patch, and can you identify the specific cell type?",
            "Have you noticed any unusual cell growth in the image section, and can you identify the specific cell type?",
            "Have you noticed any unusual cell growth in the image section, and can you identify the specific cell type?",
            "Have you identified any unusual cell growth within the image patch, and if so, can you specify the cell type?",
            "Have you noticed any unusual cell growth within the image patch, and if so, can you identify the specific cell type?",
            "Have you identified any unusual cell growth in the image section, and can you specify the cell type involved?",
            "Have you noticed any unusual cell growth within the image patch, and if so, can you identify the specific cell type?",
            "Have you noticed any unusual cell growth within the image patch, and can you identify the specific cell type?",
            "Have you noticed any unusual growth of cells, and if so, can you identify the specific cell type?",
            "Have you noticed any unusual cellular growth within the image section, and can you identify the specific cell type involved?",
        ]

        self.text_repo["a_221"] = [
            "The picture segment exhibits an atypical increase in Plasma cell numbers",
            "An unusual expansion of Plasma cells is present within the visualized area",
            "The image segment displays an atypical increase in the number of Plasma cells",
            "In the image section, there is an irregular expansion of Plasma cells present",
            "The picture segment displays an atypical increase in the number of Plasma cells",
            "An unusual expansion of Plasma cells is observed in the image portion",
            "The picture segment exhibits an unusual increase in the number of Plasma cells",
            "An atypical multiplication of Plasma cells is evident in the image section",
            "The picture segment exhibits an atypical increase in Plasma cell population",
            "An irregular expansion of Plasma cells is observed in the visual snippet",
        ]
        self.text_repo["q_221"] = [
            "As a pathologist, would you be able to determine the possible blood cancer from the image patch provided?",
            "Can you pinpoint the probable illness depicted in this image patch, acting as a pathologist?",
            "Could you discern the potential condition represented in the image patch?",
            "Can you determine the possible illness depicted in this image section?",
            "Would you be able to recognize the probable disease present in the visual fragment as a pathologist?",
            "Can you pinpoint the potential disorder shown in the image piece?",
            "Can you determine the possible ailment depicted in this image patch?",
            "Can you pinpoint the likely condition shown in the image patch?",
            "Are you able to recognize the probable illness in the image patch?",
            "Can you determine the possible illness from the provided image patch?",
            "Are you able to recognize the likely condition depicted in this image patch?",
            "Would you be able to diagnose the probable ailment in the image patch?",
            "Can you determine the possible illness depicted in this image patch, as a pathologist would?",
            "As a pathologist, are you able to pinpoint the likely ailment present in the image section?",
            "Could you recognize the probable condition displayed in the image fragment?",
        ]
        self.text_repo["a_2211"] = [
            "In this image patch, there is evidence suggestive of multiple myeloma, as a pathologist might identify",
            "The visual representation in the image patch indicates the potential presence of multiple myeloma, similar to a pathologist's determination",
            "The picture segment displays a sign of multiple myeloma",
            "In the provided image section, there is an evidence of multiple myeloma",
            "The provided image patch reveals a sign of multiple myeloma",
            "An indication of multiple myeloma is displayed in the image patch",
            "The image patch displays signs suggestive of multiple myeloma",
            "Multiple myeloma is indicated by the image patch presented",
            "The image patch displays a sign of multiple myeloma",
            "In the image patch, there is an indication of multiple myeloma being present",
        ]

        self.text_repo["a_23"] = [
            "The quality of the image patch is inadequate, making it impossible to determine the presence of any irregularities that could indicate a medical condition",
            "Due to the low quality of the image patch, it is unclear if any irregularities suggesting a potential medical issue can be identified",
            "The quality of the image patch is subpar, making it impossible to determine the presence of any irregularities",
            "Due to the low-quality image patch, it is unclear if there are any abnormalities present or not",
            "The quality of the image patch is inadequate, making it impossible to determine the presence of any irregularities",
            "Due to the low-quality image patch, it is unclear if any irregularities exist that may indicate a disease",
            "The quality of the image patch is inadequate, making it difficult to determine the presence of any irregularities",
            "Due to the low quality of the image patch, it is unclear if there are any irregularities that may indicate a medical condition",
            "The quality of the image patch is insufficient, making it impossible to determine the presence of any irregularities",
            "Due to the low-quality image patch, it is unclear if any irregularities exist that may indicate a disease",
        ]
        self.text_repo["q_23"] = [
            "Have you noticed any unusual cell growth within the image patch, and can you identify the specific cell type?",
            "Have you noticed any unusual cell growth in the image section, and can you identify the specific cell type?",
            "Have you noticed any unusual cell growth in the image section, and can you identify the specific cell type?",
            "Have you identified any unusual cell growth within the image patch, and if so, can you specify the cell type?",
            "Have you noticed any unusual cell growth within the image patch, and if so, can you identify the specific cell type?",
            "Have you identified any unusual cell growth in the image section, and can you specify the cell type involved?",
            "Have you noticed any unusual cell growth within the image patch, and if so, can you identify the specific cell type?",
            "Have you noticed any unusual cell growth within the image patch, and can you identify the specific cell type?",
            "Have you noticed any unusual growth of cells, and if so, can you identify the specific cell type?",
            "Have you noticed any unusual cellular growth within the image section, and can you identify the specific cell type involved?",
        ]
        self.text_repo["a_231"] = [
            "The quality of the image patch is insufficient, and there are not enough bone marrow cells visible to make a definitive determination",
            "Due to the low-quality image patch, an adequate number of bone marrow cells cannot be observed to reach a conclusion",
            "The quality of the image patch is insufficient, and there are not enough bone marrow cells present to make a definitive conclusion",
            "Due to the low-quality image patch, there are not an adequate number of bone marrow cells available to determine a conclusive result",
            "The quality of the image patch is insufficient, and there are not enough bone marrow cells available to make a definitive conclusion",
            "Due to the subpar quality of the image patch, an adequate number of bone marrow cells cannot be observed to determine a conclusive result",
            "The quality of the image patch is inadequate, and there are not sufficient bone marrow cells to make a definitive diagnosis",
            "Due to the subpar quality of the image patch, it is impossible to determine a conclusion as there are an insufficient number of bone marrow cells present",
            "The quality of the image patch is inadequate, lacking sufficient bone marrow cells to make a definitive determination",
            "Insufficient bone marrow cells are visible in the low-quality image patch, preventing a conclusive assessment",
        ]
        self.text_repo["q_231"] = [
            "As a pathologist, would you be able to determine the possible blood cancer from the image patch provided?",
            "Can you pinpoint the probable illness depicted in this image patch, acting as a pathologist?",
            "Could you discern the potential condition represented in the image patch?",
            "Can you determine the possible illness depicted in this image section?",
            "Would you be able to recognize the probable disease present in the visual fragment as a pathologist?",
            "Can you pinpoint the potential disorder shown in the image piece?",
            "Can you determine the possible ailment depicted in this image patch?",
            "Can you pinpoint the likely condition shown in the image patch?",
            "Are you able to recognize the probable illness in the image patch?",
            "Can you determine the possible illness from the provided image patch?",
            "Are you able to recognize the likely condition depicted in this image patch?",
            "Would you be able to diagnose the probable ailment in the image patch?",
            "Can you determine the possible illness depicted in this image patch, as a pathologist would?",
            "As a pathologist, are you able to pinpoint the likely ailment present in the image section?",
            "Could you recognize the probable condition displayed in the image fragment?",
        ]
        self.text_repo["a_2311"] = [
            "The picture segment is of low quality, making it unsuitable for accurate medical diagnosis",
            "Due to the substandard quality of the image section, it cannot be reliably used for identifying the likely medical condition",
            "The quality of the image patch is inadequate, making it unsuitable for diagnostic purposes",
            "The image patch's poor quality renders it inappropriate for determining a potential disorder",
            "The quality of the image patch is inadequate, rendering it unsuitable for accurate medical diagnosis",
            "Due to the substandard quality of the image patch, it cannot be reliably used for diagnostic purposes",
            "The provided image patch is of inadequate quality, rendering it unsuitable for accurate medical diagnosis",
            "Due to the low quality of the image patch, it cannot be reliably used for diagnostic purposes in identifying potential blood cancer",
            "The quality of the image patch is inadequate, making it unsuitable for accurate medical diagnosis",
            "The image patch's low quality renders it inappropriate for reliable diagnostic purposes",
        ]

        self.text_repo["a_24"] = [
            "The image patch appears to be normal, with no observable irregularities present",
            "No irregularities can be detected in the image patch, indicating a normal appearance",
            "The image patch appears to be normal, with no observable irregularities detected",
            "No irregularities can be identified in the image patch, suggesting a normal appearance",
            "The image patch appears normal, with no observable irregularities present",
            "No irregularities can be detected in the image patch, as it seems to be normal",
            "The image patch appears normal, with no visible irregularities detected",
            "No abnormalities can be observed in the image patch, as it seems to be normal",
            "The visual segment appears to be normal, with no noticeable abnormalities detected",
            "No irregularities can be discerned in the image patch, indicating a normal appearance",
        ]

        self.text_repo["q_24"] = [
            "Have you noticed any unusual cell growth within the image patch, and can you identify the specific cell type?",
            "Have you noticed any unusual cell growth in the image section, and can you identify the specific cell type?",
            "Have you noticed any unusual cell growth in the image section, and can you identify the specific cell type?",
            "Have you identified any unusual cell growth within the image patch, and if so, can you specify the cell type?",
            "Have you noticed any unusual cell growth within the image patch, and if so, can you identify the specific cell type?",
            "Have you identified any unusual cell growth in the image section, and can you specify the cell type involved?",
            "Have you noticed any unusual cell growth within the image patch, and if so, can you identify the specific cell type?",
            "Have you noticed any unusual cell growth within the image patch, and can you identify the specific cell type?",
            "Have you noticed any unusual growth of cells, and if so, can you identify the specific cell type?",
            "Have you noticed any unusual cellular growth within the image section, and can you identify the specific cell type involved?",
        ]
        self.text_repo["a_241"] = [
            "Based on my observation, the image patch does not display any abnormal cellular growth",
            "Upon examination, I cannot detect any atypical cell proliferation within the image patch",
            "From my analysis, the image patch does not exhibit any atypical cell proliferation",
            "Based on my observation, the image section appears to be devoid of any abnormal cellular expansion",
            "To the best of my knowledge, the image patch does not display any atypical cellular proliferation",
            "Upon examination, I have not observed any abnormal cell expansion within the image patch",
            "Based on my observation, the image patch does not reveal any abnormal cellular proliferation",
            "Upon examination, the image patch appears to be free from any atypical cell development",
            "Based on my analysis, the image patch does not display any atypical cellular growth",
            "Upon examination, I cannot detect any abnormal cell proliferation within the image section",
        ]
        self.text_repo["q_241"] = [
            "As a pathologist, would you be able to determine the possible blood cancer from the image patch provided?",
            "Can you pinpoint the probable illness depicted in this image patch, acting as a pathologist?",
            "Could you discern the potential condition represented in the image patch?",
            "Can you determine the possible illness depicted in this image section?",
            "Would you be able to recognize the probable disease present in the visual fragment as a pathologist?",
            "Can you pinpoint the potential disorder shown in the image piece?",
            "Can you determine the possible ailment depicted in this image patch?",
            "Can you pinpoint the likely condition shown in the image patch?",
            "Are you able to recognize the probable illness in the image patch?",
            "Can you determine the possible illness from the provided image patch?",
            "Are you able to recognize the likely condition depicted in this image patch?",
            "Would you be able to diagnose the probable ailment in the image patch?",
            "Can you determine the possible illness depicted in this image patch, as a pathologist would?",
            "As a pathologist, are you able to pinpoint the likely ailment present in the image section?",
            "Could you recognize the probable condition displayed in the image fragment?",
        ]
        self.text_repo["a_2411"] = [
            "The visual segment reveals no evidence of blood cancer, and the individual is in good health",
            "There is no indication of blood cancer in the picture section, and the person appears to be healthy",
            "The visual representation reveals no evidence of blood cancer, signifying the patient's well-being",
            "There is an absence of blood cancer indications in the image patch, confirming the patient's healthy state",
            "The image patch reveals no evidence of blood cancer, confirming the patient's healthy condition",
            "There is an absence of blood cancer indications in the image patch, signifying the patient's good health",
            "The image patch reveals no evidence of blood cancer, suggesting that the patient is in good health",
            "There is no indication of blood cancer in the image patch, which implies that the patient is healthy",
            "The provided image patch reveals no evidence of blood cancer, and the patient appears to be in good health",
            "There is no indication of blood cancer in the image patch, suggesting that the patient is healthy",
        ]


### mian function
if __name__ == "__main__":
    assert 1 == 2, "this is a test"
