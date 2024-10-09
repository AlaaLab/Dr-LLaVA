import argparse
import json
import os
from torch import Tensor
from transformers.utils.generic import ModelOutput
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as score, accuracy_score

class RewardModelOutput(ModelOutput):
    rewards: Tensor = None

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
    """
    Rule based reward model for ACUTE MYOCARDIAL INFACTION DETECTION
    For different user cases, the reward model will return different rewards based on the input sentences
    Please edit the below functions to match the user case
    """

    def __init__(self, align_lambda=2):
        self.align_lambda = align_lambda
        self.knowledge = {
            "NORMAL": [False, False, False, "normal"], #All paths possible, no alignment reward thus
            "STEMI": [True, False, True, "stemi"], #stemi & must have ST_elevation
            "NSTEMI": [False,True, True, "nstemi"], #nstemi & must have no ST_elevation
        }

    def _get_method(self, index, length):
        """
        Returns the method associated with the given category index.
        """
        if length ==4:
            method_dict = {
                1: self._check_st_elevation,
                2: self._check_st_depression_or_t_wave,
                3: self._ordering_troponin_test,
                4: self._diagnosis,
            }
        else:
            method_dict = {
                1: self._check_st_elevation,
                2: self._ordering_troponin_test,
                3: self._diagnosis,
            }
        return method_dict.get(index, None)  # Returns None if index is not found

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
            sentences, ["stemi", "st elevation", 'st elevation myocardial infarction']
         ) and no_exist_in_sentence(
            sentences, [" not ", " no ", " absen"]
         ):
            return "stemi"
        elif exist_in_sentence(
            sentences, ["nstemi", "no st elevation", "non-st elevation myocardial infarction", "non st elevation myocardial infarction"]
        ):
            return "nstemi"
        else:
            return "normal"

    def calculate_reward(
        self,
        sentences,
        return_dict=True,
        device=None,
        ref_answers=None,
        categories=None,
    ):
        """
        This function is used to calculate the accumulated reward for a series of sentences
        """
        outcomes = []
        gt = []
        # sort the sentences and ref_answers by categories

        for sentence, category in zip(sentences, categories):
            method = self._get_method(category, len(categories))
            if not method:
                raise ValueError(f"No method found for category {category}")
            outcome = method(sentence)
            outcomes.append(outcome)

        for ref_answer, category in zip(ref_answers, categories):
            method = self._get_method(category, len(categories))
            if not method:
                raise ValueError(f"No method found for category {category}")
            gt_result = method(ref_answer)
            gt.append(gt_result)

        correct_bonus = [
            1 if x == y else -0.5 if x == "no match" else 0
            for x, y in zip(outcomes, gt)
        ]

        align_bonus = []
        if outcomes[-1] in ["stemi", "nstemi", "normal"]:
            if (outcomes[-1] == "stemi") and (outcomes[0] == True):
                align_bonus.append(1)
            elif (outcomes[-1] == "nstemi") and (outcomes[0] == False):
                align_bonus.append(1)
            else:
                align_bonus.append(0)

            # for i in range(1, len(categories)):
            #     res = [
            #         [self.knowledge[key][categories[x] - 1] for x in [i - 1, i]]
            #         for key in self.knowledge.keys()
            #     ]
            #     if outcomes[i - 1 : i + 1] in res:
            #         align_bonus.append(1)
            #     else:
            #         align_bonus.append(0)
        else:
            align_bonus.append(0)
        # length_bonus is calculated if the length of the outcomes is the same as the length of the ref_answers has more than 10 letters difference
        length_bonus = [
            (
                -abs(len(x) - len(y)) / 10
                if (abs(len(x) - len(y)) / 10) > 1
                else 0
            )
            for x, y in zip(outcomes, gt)
        ]
        # calculate the total bonus
        bonus = sum(correct_bonus) + self.align_lambda*sum(align_bonus) + sum(length_bonus)
        return RewardModelOutput(rewards=bonus) if return_dict else (None,)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--output-result', type=str)
    parser.add_argument('--options', type=list, default=["Yes", "No"])
    return parser.parse_args()


rm = RewardModel_ACS()

if __name__ == "__main__":
    args = get_args()

    base_dir = args.base_dir
    problems = json.load(open(os.path.join(base_dir, "test_conversations.json")))
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


        pred_diagnose = 'normal'
        if preds[3]['text'] == 'Yes':
            if rm._check_st_elevation(preds[0]['text']):
                pred_diagnose = 'stemi'
            else:
                pred_diagnose = 'nstemi'
        y_pred[-1][-1] = pred_diagnose

        y_true.append([
            rm._check_st_elevation(convs[1]['value']),
            rm._check_st_depression_or_t_wave(convs[3]['value']),
            rm._ordering_troponin_test(convs[5]['value']),
            rm._diagnosis(convs[7]['value']),
        ])

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    for i in range(4):
        precision, recall, fscore, support = score(y_true[:,i], y_pred[:,i], average='macro')

        print('accuracy: {}'.format(accuracy_score(y_true[:,i], y_pred[:,i])))
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))

    precision, recall, fscore, support = score(y_true.flatten(), y_pred.flatten(), average='macro')
    print('accuracy: {}'.format(accuracy_score(y_true.flatten(), y_pred.flatten())))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

    y_true = [ ''.join(y.tolist()) for y in y_true ]
    y_pred = [ ''.join(y.tolist()) for y in y_pred ]

    precision, recall, fscore, support = score(y_true, y_pred, average='macro')
    print('accuracy: {}'.format(accuracy_score(y_true, y_pred)))
    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))
