"""
Model utils
"""
import os
import json
import numpy as np
from itertools import groupby
import torch
import torch.nn.functional as F

def clean_text(inp):
    """

    :param inp:
    :return:
    """
    def clean(txt):
        return "\n".join(
            line.replace("•", "").replace("-", "").replace("*", "").replace("#", " ")
            for line in txt.split("\n")
        )

    if isinstance(inp, list):
        return_out = ",".join([clean(string) for string in inp])
    elif isinstance(inp, str):
        return_out = clean(inp)

    return return_out

def read_json_data(filepath):
    """

    :param filepath:
    :return:
    """
    if not isinstance(filepath, list):
        filepath = [filepath]

    return_json_content_dict = dict()
    for path in filepath:
        if os.path.isfile(path) and os.path.splitext(path)[-1].lower()==".json":
            with open(path) as json_in:
                json_data = json.load(json_in)
        else:
            json_data = None
        return_json_content_dict[path] = json_data
    return return_json_content_dict


def get_word_proba(emmision_matrix, transition_matrix, decoded_out, o_index):
    """

    :param emmision_matrix:
    :param transition_matrix:
    :param decoded_out:
    :param o_index:
    :return:
    """
    out_proba = []
    softmax_scores = []

    for k in range(emmision_matrix.size(0)):
        out_proba_this = []
        this_softmax_score = []
        for kj in range(emmision_matrix[k].size(0)):
            if kj == 0:
                prev = o_index
            curr = decoded_out[k][kj]
            sfm_score = F.softmax(emmision_matrix[k][kj]+transition_matrix[prev])
            this_softmax_score.append(sfm_score)
            out_proba_this.append(sfm_score[curr])
            prev = curr

        out_proba.append(torch.Tensor(out_proba_this))
        softmax_scores.append(torch.stack(this_softmax_score))

    out_proba = torch.stack(out_proba)
    softmax_scores = torch.stack(softmax_scores)

    return out_proba, softmax_scores

def get_entities_values_joint_probas(result:list=[], sentence:list=[], proba:list=[], log_score=False, add_factor=.5, restrict_if_no_begining=True):
    """

    :param result:
    :param sentence:
    :param proba:
    :return:
    """
    entity_values = []
    entities = []
    probas = []
    this_entity_value_list = []
    this_proba = 0
    found_begining = False

    for ind, label in enumerate(result):
        if label == "O":
            this_entity_value_list = []
            entity_value_finish = None
            this_proba = 0
            found_begining = False

        elif label != "O" and "-" in label:
            if "-B" in label:
                found_begining = True

            if found_begining or (not restrict_if_no_begining):
                this_entity_value_list.append(sentence[ind])
                if log_score:
                    this_proba += -1*np.log(proba[ind])
                else:
                    this_proba += proba[ind]

                if (ind < len(result)-1 and result[ind + 1] == "O") or ind == len(sentence)-1:
                    entity_value_finish = True
                    this_entity = label[:label.find("-")]
                else:
                    entity_value_finish = False

        if len(this_entity_value_list) > 0 and entity_value_finish == True:
            entities.append(this_entity)
            entity_values.append(" ".join(this_entity_value_list))
            probas.append(this_proba/(len(this_entity_value_list)+add_factor))

    return tuple(zip(entities, entity_values, probas))


def get_one_value_each_entity(final_out_list):
    """

    :param final_out_list:
    :return:
    """
    # Loop through list of nested tuples
    return_dict_list = []
    for l in range(len(final_out_list)):
        this_final_out = final_out_list[l]
        return_ner_dict = {label[0]: [] for label in this_final_out}
        return_proba_dict = {label[0]: [] for label in this_final_out}
        # Loop thorugh each tuple in each sentence i.e. each entity
        # and create list of outcomes

        for tupe in this_final_out:
            return_ner_dict[tupe[0]].append(tupe[1])
            return_proba_dict[tupe[0]].append(tupe[2])

        # Finally choose the final output based on probability
        return_final_dict = dict()
        for key, val in return_proba_dict.items():

            max_prob_index = np.argmax(np.array(val))
            return_final_dict[key]=(return_ner_dict[key][max_prob_index], val[max_prob_index])

        return_dict_list.append(return_final_dict)

    return return_dict_list
