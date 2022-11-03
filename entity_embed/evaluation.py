import csv
import json
from .indexes import ANNEntityIndex
from .data_utils import utils
import pandas as pd


def pair_entity_ratio(found_pair_set_len, entity_count):
    return found_pair_set_len / entity_count


def precision_and_recall(found_pair_set, pos_pair_set, neg_pair_set=None):
    # if a neg_pair_set is provided,
    # consider the "universe" to be only the what's inside pos_pair_set and neg_pair_set,
    # because this means a previous blocking was applied
    if neg_pair_set is not None:
        found_pair_set = found_pair_set & (pos_pair_set | neg_pair_set)

    true_positives = found_pair_set & pos_pair_set
    false_positives = found_pair_set - pos_pair_set
    if true_positives:
        precision = len(true_positives) / (len(true_positives) + len(false_positives))
    else:
        precision = 0.0
    recall = len(true_positives) / len(pos_pair_set)
    return precision, recall


def f1_score(precision, recall):
    if precision or recall:
        return (2 * precision * recall) / (precision + recall)
    else:
        return 0.0


def evaluate_output_json(
    unlabeled_csv_filepath, output_json_filepath, pos_pair_json_filepath, csv_encoding="utf-8"
):
    with open(
        unlabeled_csv_filepath, "r", newline="", encoding=csv_encoding
    ) as record_dict_csv_file:
        record_count = sum(1 for __ in csv.DictReader(record_dict_csv_file))

    with open(output_json_filepath, "r") as f:
        found_pair_set = json.load(f)
    found_pair_set = set(tuple(t) for t in found_pair_set)

    with open(pos_pair_json_filepath, "r") as f:
        pos_pair_set = json.load(f)
    pos_pair_set = set(tuple(t) for t in pos_pair_set)

    precision, recall = precision_and_recall(found_pair_set, pos_pair_set)
    return (
        precision,
        recall,
        f1_score(precision, recall),
        pair_entity_ratio(len(found_pair_set), record_count),
    )


class EmbeddingEvaluator:
    def __init__(self, record_dict, vector_dict, cluster_field='cluster_id'):
        self.record_dict = record_dict
        self.cluster_field = cluster_field
        embedding_size = len(next(iter(vector_dict.values())))
        self.ann_index = ANNEntityIndex(embedding_size)
        self.ann_index.insert_vector_dict(vector_dict)
        self.ann_index.build()

    def evaluate(self, k, sim_thresholds):
        cluster_dict = utils.record_dict_to_cluster_dict(self.record_dict, self.cluster_field)
        pos_pair_set = utils.cluster_dict_to_id_pairs(cluster_dict)
        results = []
        for sim_threshold in sim_thresholds:
            found_pair_set = self.ann_index.search_pairs(k, sim_threshold)
            precision, recall = precision_and_recall(found_pair_set, pos_pair_set)
            results.append((sim_threshold, precision, recall, f1_score(precision, recall)))
        return pd.DataFrame(results, columns=['threshold', 'precision', 'recall', 'f1_score'])
