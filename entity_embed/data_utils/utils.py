import itertools
import logging
import os
import random
from collections import Counter, defaultdict
from importlib import import_module

from ordered_set import OrderedSet

from .union_find import UnionFind

logger = logging.getLogger(__name__)


def Enumerator(start=0, initial=()):
    return defaultdict(itertools.count(start).__next__, initial)


def row_dict_to_cluster_dict(row_dict, cluster_attr):
    cluster_dict = defaultdict(list)
    for id_, row in row_dict.items():
        cluster_dict[row[cluster_attr]].append(id_)

    # must use sorted to always have smaller id on left of pair tuple
    for c in cluster_dict.values():
        c.sort()
    return cluster_dict


def cluster_dict_to_id_pairs(cluster_dict):
    return set(
        pair
        for cluster_id_list in cluster_dict.values()
        # must use sorted to always have smaller id on left of pair tuple
        for pair in itertools.combinations(sorted(cluster_id_list), 2)
    )


def count_cluster_dict_pairs(cluster_dict):
    return sum(
        (len(cluster_id_list) * (len(cluster_id_list) - 1)) // 2
        for cluster_id_list in cluster_dict.values()
    )


def split_clusters(
    cluster_dict, train_len, valid_len, test_len, random_seed, only_plural_clusters=True
):
    rnd = random.Random(random_seed)
    if only_plural_clusters:
        # consider only clusters that have more than 1 entity for train and valid
        all_cluster_id_set = OrderedSet(
            cluster_id
            for cluster_id, cluster_id_list in cluster_dict.items()
            if len(cluster_id_list) > 1
        )
    else:
        all_cluster_id_set = cluster_dict.keys()

    if train_len + valid_len + test_len < len(all_cluster_id_set):
        logger.warning(
            f"(train_len + valid_len + test_len)={train_len + valid_len + test_len} "
            f"is less than len(all_cluster_id_set)={len(all_cluster_id_set)}"
        )

    train_cluster_id_set = OrderedSet(rnd.sample(all_cluster_id_set, train_len))
    all_minus_train_cluster_id_set = all_cluster_id_set - train_cluster_id_set
    valid_cluster_id_set = OrderedSet(rnd.sample(all_minus_train_cluster_id_set, valid_len))
    test_cluster_id_set = all_minus_train_cluster_id_set - valid_cluster_id_set
    if test_len < len(test_cluster_id_set):
        test_cluster_id_set = OrderedSet(rnd.sample(test_cluster_id_set, test_len))

    assert train_cluster_id_set.isdisjoint(valid_cluster_id_set)
    assert train_cluster_id_set.isdisjoint(test_cluster_id_set)
    assert valid_cluster_id_set.isdisjoint(test_cluster_id_set)

    train_cluster_dict = {
        cluster_id: cluster_dict[cluster_id] for cluster_id in train_cluster_id_set
    }
    valid_cluster_dict = {
        cluster_id: cluster_dict[cluster_id] for cluster_id in valid_cluster_id_set
    }
    test_cluster_dict = {cluster_id: cluster_dict[cluster_id] for cluster_id in test_cluster_id_set}
    return train_cluster_dict, valid_cluster_dict, test_cluster_dict


def cluster_dict_to_filtered_row_dict(row_dict, cluster_dict):
    return {
        id_: row_dict[id_] for cluster_id_list in cluster_dict.values() for id_ in cluster_id_list
    }


def cluster_dicts_to_row_dicts(row_dict, train_cluster_dict, valid_cluster_dict, test_cluster_dict):
    train_row_dict = cluster_dict_to_filtered_row_dict(row_dict, train_cluster_dict)
    valid_row_dict = cluster_dict_to_filtered_row_dict(row_dict, valid_cluster_dict)
    test_row_dict = cluster_dict_to_filtered_row_dict(row_dict, test_cluster_dict)
    return train_row_dict, valid_row_dict, test_row_dict


def dict_filtered_from_id_set(d, id_set):
    return {id_: row for id_, row in d.items() if id_ in id_set}


def separate_dict_left_right(d, left_id_set, right_id_set):
    return (
        dict_filtered_from_id_set(d, left_id_set),
        dict_filtered_from_id_set(d, right_id_set),
    )


def compute_max_str_len(attr_val_gen, is_multitoken, tokenizer):
    actual_max_str_len = 0
    for attr_val in attr_val_gen:
        if not is_multitoken:
            str_len = len(attr_val)
        else:
            tokens = tokenizer(attr_val)
            if tokens:
                str_len = max(len(v) for v in tokens)
            else:
                str_len = -1
        actual_max_str_len = max(str_len, actual_max_str_len)

    # Ensure max_str_len is even to enable pooling later
    if actual_max_str_len % 2 != 0:
        logger.info(
            f"actual_max_str_len={actual_max_str_len} must be even to enable NN pooling. "
            f"Updating to {actual_max_str_len + 1}"
        )
        actual_max_str_len += 1

    return actual_max_str_len


def compute_vocab_counter(attr_val_gen, tokenizer):
    vocab_counter = Counter()
    for attr_val in attr_val_gen:
        tokens = tokenizer(attr_val)
        vocab_counter.update(tokens)
    return vocab_counter


def id_pairs_to_cluster_mapping_and_dict(id_pairs):
    uf = UnionFind()
    uf.union_pairs(id_pairs)
    cluster_dict = uf.component_dict()
    # must use sorted to always have smaller id on left of pair tuple
    for c in cluster_dict.values():
        c.sort()
    # must be called after component_dict, because of find calls
    cluster_mapping = uf.parents
    return cluster_mapping, cluster_dict


def import_function(function_dotted_path):
    module_dotted_path, function_name = function_dotted_path.rsplit(".", 1)
    module = import_module(module_dotted_path)
    return getattr(module, function_name)


def build_loader_kwargs(**kwargs):
    num_workers = kwargs.get("num_workers") or os.cpu_count()
    multiprocessing_context = kwargs.get("multiprocessing_context") or "fork"
    return {
        "num_workers": num_workers,
        "multiprocessing_context": multiprocessing_context,
    }
