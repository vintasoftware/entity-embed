import csv
import json
import random
from .indexes import ANNEntityIndex
from .data_utils import utils
import pandas as pd
import logging

logger = logging.getLogger(__name__)

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
    def __init__(self, record_dict, vector_dict, cluster_field="cluster_id"):
        self.record_dict = record_dict
        self.cluster_field = cluster_field
        embedding_size = len(next(iter(vector_dict.values())))
        logging.info("Building index...")
        self.ann_index = ANNEntityIndex(embedding_size)
        self.ann_index.insert_vector_dict(vector_dict)
        self.ann_index.build()
        logging.info("Index built! Getting cluster dict...")
        self.cluster_dict = utils.record_dict_to_cluster_dict(self.record_dict, self.cluster_field)
        logging.info("Getting positive pairs...")
        self.pos_pair_set = utils.cluster_dict_to_id_pairs(self.cluster_dict)

    def evaluate(self, k, sim_thresholds, query_ids=None, get_missing_pair_set=False):
        """
        params:
        k: int: number of nearest neighbours to retrieve
        sim_thresholds: list of floats in the range [0,1]:
        query_ids: list or set of ids that must be keys in self.vector_dict and self.record_dict. Indicates
            which ids to find pairs for. If None, use all record ids as query ids

        returns: pandas DataFrame of results, with one row for each threshold
        """
        if query_ids is None:
            logging.info(f"Using all {len(self.record_dict)} records to query for neighbours")
            pos_pair_subset = self.pos_pair_set
        else:
            query_ids = set(query_ids)
            logging.info(f"Using subset of {len(query_ids)} query IDs")
            pos_pair_subset = {
                pair for pair in self.pos_pair_set if pair[0] in query_ids or pair[1] in query_ids
            }
        results = []
        for sim_threshold in sim_thresholds:
            found_pair_set = self.ann_index.search_pairs(
                k, sim_threshold, query_id_subset=query_ids
            )
            precision, recall = precision_and_recall(found_pair_set, pos_pair_subset)
            results.append((sim_threshold, precision, recall, f1_score(precision, recall)))
            if get_missing_pair_set & (sim_threshold == min(sim_thresholds)):
                self.missing_pair_set = pos_pair_subset - found_pair_set
                id_to_name_map = {k: v["merchant_name"] for k, v in self.record_dict.items()}
                self.missing_pair_name_set = set(
                    map(
                        lambda x: (id_to_name_map[x[0]], id_to_name_map[x[1]]),
                        self.missing_pair_set,
                    )
                )

        return pd.DataFrame(results, columns=["threshold", "precision", "recall", "f1_score"])
