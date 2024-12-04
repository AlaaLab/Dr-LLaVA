import argparse
import json
import os
import numpy as np
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support as score, accuracy_score

def exist_in_sentence(sentence, keywords):
    # check if keywords exist in the sentence
    # sentence: a string
    # keywords: a list of string
    for keyword in keywords:
        if keyword in sentence:
            return True
    return False


def no_exist_in_sentence(sentence, keywords):
    # check if keywords exist in the sentence
    # sentence: a string
    # keywords: a list of string
    for keyword in keywords:
        if keyword in sentence:
            return False
    return True

class RewardModel_ACS:
    # write 4 function to check if each answer is cosistent with the question
    # the 4 question cover 4 aspects
    # check st elevation; check st depression or t wave inversion; ordering troponin test; Diagnosis
    def _check_st_elevation(self, sentences):
        sentences = sentences.lower()
        if exist_in_sentence(
            sentences, ["yes", "indeed", "shows", "indicate", "evidence", "present"]
        ) and no_exist_in_sentence(
            sentences, [" not ", " no "]
        ):
            return True
        elif exist_in_sentence(
            sentences, ["cannot", "not", "no"]
        ):
            return False
        else:
            return "no match"

    def _check_st_depression_or_t_wave(self, sentences):
        sentences = sentences.lower()
        if exist_in_sentence(
            sentences, ["yes", "indeed", "shows", "indicate", "evidence", "present"]
        ) and no_exist_in_sentence(
            sentences, [" not ", " no "]
        ):
            return True
        elif exist_in_sentence(
            sentences, ["cannot", "not", "no"]
        ):
            return False
        else:
            return "no match"

    def _ordering_troponin_test(self, sentences):
        sentences = sentences.lower()
        if exist_in_sentence(
            sentences, ["yes", "indeed", "should", "ordering", "necessary", "advis"]
        ) and no_exist_in_sentence(
            sentences, [" not ", " no "]
        ):
            return True
        elif exist_in_sentence(
            sentences, ["cannot", "not", "no", "unnecessary"]
        ):
            return False
        else:
            return "no match"

    def _diagnosis(self, sentences):
        sentences = sentences.lower()
        if exist_in_sentence(
            sentences, ["stemi", 'st elevation myocardial infarction']
         ) and no_exist_in_sentence(
            sentences, [" not ", " no ", " absen"]
         ) or (exist_in_sentence(
            sentences, ['myocardial infarction', 'heart attack', 'acute mi', 'mi', 'acs', 'acute coronary syndrome']
         ) and exist_in_sentence(
            sentences, ['st elevation', 'ste']
         ) and no_exist_in_sentence(
            sentences, [" not ", " no ", " absen"]
         )):
            return "stemi"
        elif exist_in_sentence(
            sentences, ["nstemi", "non-st elevation myocardial infarction", "non st elevation myocardial infarction"]
        ) or (exist_in_sentence(
            sentences, ['myocardial infarction', 'heart attack', 'acute mi', 'mi', 'acs', 'acute coronary syndrome']
         ) and exist_in_sentence(
            sentences, ['non st elevation', 'no st elevation', 'no ste']
         )):
            return "nstemi"
        elif exist_in_sentence(
            sentences, ["not", "no",'no indication', 'no heart attack']
        ) or (exist_in_sentence(
            sentences, ["healthy", 'without disease']
         ) and no_exist_in_sentence(
            sentences, [" not ", " no "]
         )):
            return "normal"
        else:
            return "no match"

rm = RewardModel_ACS()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--output-result', type=str)
    parser.add_argument('--options', type=list, default=["Yes", "No"])
    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()

    base_dir = args.base_dir
    problems = json.load(open(os.path.join(base_dir, "/home/mac/wday/Dr-LLaVA/data/test_conversations_with_preds_simple.json")))
    predictions = [json.loads(line) for line in open(args.result_file)]

    y_pred = []
    y_true = []

    for i, prob in enumerate(problems):
        preds = predictions[i*4:(i+1)*4]
        convs = prob['conversations']

        y_pred.append([
            rm._check_st_elevation(preds[0]['text']),
            rm._check_st_depression_or_t_wave(preds[1]['text']),
            rm._ordering_troponin_test(preds[2]['text']),
            rm._diagnosis(preds[3]['text']),
        ])

        y_true.append([
            rm._check_st_elevation(convs[1]['value']),
            rm._check_st_depression_or_t_wave(convs[3]['value']),
            rm._ordering_troponin_test(convs[5]['value']),
            rm._diagnosis(convs[7]['value']),
        ])

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    for i in range(4):
        print('---')
        print(f'Q_{i+1}')

        precision, recall, fscore, support = score(y_true[:,i], y_pred[:,i], average='macro')

        print('accuracy: {}'.format(accuracy_score(y_true[:,i], y_pred[:,i])))
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('f1-score: {}'.format(fscore))

        x = Counter(y_pred[:,i].tolist())
        print(x)


    print('---')
    print('A_Q')

    precision, recall, fscore, support = score(y_true.flatten(), y_pred.flatten(), average='macro')
    print('accuracy: {}'.format(accuracy_score(y_true.flatten(), y_pred.flatten())))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('f1-score: {}'.format(fscore))

    print('---')
    print('A_C')

    y_true = [ ''.join(y.tolist()) for y in y_true ]
    y_pred = [ ''.join(y.tolist()) for y in y_pred ]

    precision, recall, fscore, support = score(y_true, y_pred, average='macro')
    print('accuracy: {}'.format(accuracy_score(y_true, y_pred)))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('f1-score: {}'.format(fscore))
