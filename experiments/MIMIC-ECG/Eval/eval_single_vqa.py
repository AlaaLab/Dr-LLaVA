import argparse
import json
import os
from sklearn.metrics import accuracy_score

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
    
    predictions = {}
    for line in open(args.result_file):
        pred = json.loads(line)
        if pred['question_id'] not in predictions:
            predictions[pred['question_id']] = []
        predictions[pred['question_id']].append(pred['text'])

    y_pred = []
    y_true = []

    for prob in problems:
        conv = prob['conversations']
        pred = predictions[prob['id']]

        y_pred.append(rm._check_st_elevation(pred[0]))
        y_true.append(rm._check_st_elevation(conv[1]['value']))
        y_pred.append(rm._check_st_depression_or_t_wave(pred[1]))
        y_true.append(rm._check_st_depression_or_t_wave(conv[3]['value']))
        y_pred.append(rm._ordering_troponin_test(pred[2]))
        y_true.append(rm._ordering_troponin_test(conv[5]['value']))
        y_pred.append(rm._diagnosis(pred[3]))
        y_true.append(rm._diagnosis(conv[7]['value']))

    print('accuracy: {}'.format(accuracy_score(y_true, y_pred)))
