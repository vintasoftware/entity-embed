import itertools
import logging
import math
import random
from collections import defaultdict

from ordered_set import OrderedSet

logger = logging.getLogger(__name__)


def row_dict_to_cluster_dict(row_dict, cluster_attr):
    cluster_dict = defaultdict(list)
    for id_, row in row_dict.items():
        cluster_dict[row[cluster_attr]].append(id_)

    # must use sorted to always have smaller id on left of pair tuple
    cluster_dict = {
        cluster_id: sorted(cluster_id_list) for cluster_id, cluster_id_list in cluster_dict.items()
    }

    return cluster_dict


def cluster_dict_to_id_pairs(cluster_dict):
    return set(
        pair
        for cluster_id_list in cluster_dict.values()
        for pair in itertools.combinations(cluster_id_list, 2)
    )


def count_cluster_dict_pairs(cluster_dict):
    return sum(
        (len(cluster_id_list) * (len(cluster_id_list) - 1)) // 2
        for cluster_id_list in cluster_dict.values()
    )


def row_dict_to_id_pairs(row_dict, cluster_attr):
    cluster_dict = row_dict_to_cluster_dict(row_dict, cluster_attr)
    return cluster_dict_to_id_pairs(cluster_dict)


def split_clusters(
    cluster_dict, train_len, valid_len, test_len, random_seed, only_plural_clusters=True
):
    rnd = random.Random(random_seed)
    if only_plural_clusters:
        # consider only clusters that have more than 1 entity for train and valid
        all_cluster_id_set = {
            cluster_id
            for cluster_id, cluster_id_list in cluster_dict.items()
            if len(cluster_id_list) > 1
        }
    else:
        all_cluster_id_set = cluster_dict.keys()

    if train_len + valid_len + test_len < len(all_cluster_id_set):
        logger.warning(
            f"{train_len + valid_len + test_len=} is less than {len(all_cluster_id_set)=}"
        )

    train_cluster_id_set = OrderedSet(rnd.sample(all_cluster_id_set, train_len))
    all_minus_train_cluster_id_set = all_cluster_id_set - train_cluster_id_set
    valid_cluster_id_set = OrderedSet(rnd.sample(all_minus_train_cluster_id_set, valid_len))
    test_cluster_id_set = OrderedSet(
        rnd.sample(all_minus_train_cluster_id_set - valid_cluster_id_set, test_len)
    )

    train_cluster_dict = {
        cluster_id: cluster_dict[cluster_id] for cluster_id in train_cluster_id_set
    }
    valid_cluster_dict = {
        cluster_id: cluster_dict[cluster_id] for cluster_id in valid_cluster_id_set
    }
    test_cluster_dict = {cluster_id: cluster_dict[cluster_id] for cluster_id in test_cluster_id_set}
    return train_cluster_dict, valid_cluster_dict, test_cluster_dict


def split_cluster_to_id_pairs(train_cluster_dict, valid_cluster_dict, test_cluster_dict):
    train_true_pair_set = cluster_dict_to_id_pairs(train_cluster_dict)
    valid_true_pair_set = cluster_dict_to_id_pairs(valid_cluster_dict)
    test_true_pair_set = cluster_dict_to_id_pairs(test_cluster_dict)
    return train_true_pair_set, valid_true_pair_set, test_true_pair_set


def cluster_dict_to_filtered_row_dict(row_dict, cluster_dict):
    return {
        id_: row_dict[id_] for cluster_id_list in cluster_dict.values() for id_ in cluster_id_list
    }


def split_clusters_to_row_dicts(
    row_dict, train_cluster_dict, valid_cluster_dict, test_cluster_dict
):
    train_row_dict = cluster_dict_to_filtered_row_dict(row_dict, train_cluster_dict)
    valid_row_dict = cluster_dict_to_filtered_row_dict(row_dict, valid_cluster_dict)
    test_row_dict = cluster_dict_to_filtered_row_dict(row_dict, test_cluster_dict)
    return train_row_dict, valid_row_dict, test_row_dict


def pair_count_to_row_count(pair_count):
    # positive solution of n for y of y = (n * (n - 1)) / 2
    # where y is pair_count and n is row_count
    return int((1 + math.sqrt(1 + 8 * pair_count)) / 2)


def compute_alphabet_and_max_str_len(attr_val_gen, is_multitoken, tokenizer):
    actual_alphabet = set()
    actual_max_str_len = 0
    for attr_val in attr_val_gen:
        actual_alphabet.update(list(attr_val))
        if not is_multitoken:
            str_len = len(attr_val)
        else:
            token_lens = [len(v) for v in tokenizer(attr_val)]
            str_len = max(token_lens) if token_lens else -1
        actual_max_str_len = max(str_len, actual_max_str_len)

    # Sort alphabet for reproducibility
    actual_alphabet = sorted(actual_alphabet)

    # Ensure max_str_len is pair to enable pooling later
    if actual_max_str_len % 2 != 0:
        logger.info(
            f"{actual_max_str_len=} must be pair to enable NN pooling. "
            f"Updating to {actual_max_str_len + 1}"
        )
        actual_max_str_len += 1

    return actual_alphabet, actual_max_str_len
