import itertools
import logging
import random
from collections import Counter, defaultdict

from ordered_set import OrderedSet

from .union_find import UnionFind

logger = logging.getLogger(__name__)


def Enumerator(start=0, initial=()):
    return defaultdict(itertools.count(start).__next__, initial)


def record_dict_to_cluster_dict(record_dict, cluster_field):
    cluster_dict = defaultdict(list)
    for id_, record in record_dict.items():
        cluster_id = record[cluster_field]
        if not isinstance(cluster_id, int):
            raise ValueError(
                "cluster_field values must always be an int, "
                f"found {type(cluster_id)} at record={record}"
            )
        cluster_dict[cluster_id].append(id_)

    # sort to always have smaller id on left of pair tuple
    for c in cluster_dict.values():
        c.sort()
    return dict(cluster_dict)  # convert to dict to avoid defaultdict


def cluster_dict_to_id_pairs(cluster_dict):
    return set(
        pair
        for cluster_id_list in cluster_dict.values()
        # sort to always have smaller id on left of pair tuple
        for pair in itertools.combinations(sorted(cluster_id_list), 2)
    )


def record_dict_to_id_pairs(record_dict, cluster_field):
    cluster_dict = record_dict_to_cluster_dict(record_dict, cluster_field)
    return cluster_dict_to_id_pairs(cluster_dict)


def count_cluster_dict_pairs(cluster_dict):
    return sum(
        (len(cluster_id_list) * (len(cluster_id_list) - 1)) // 2
        for cluster_id_list in cluster_dict.values()
    )


def _split_cluster_dict(cluster_dict, train_len, valid_len, random_seed):
    rnd = random.Random(random_seed)

    cluster_id_set = OrderedSet(cluster_dict.keys())  # ensure deterministic order
    train_cluster_id_set = OrderedSet(rnd.sample(list(cluster_id_set), train_len))
    all_minus_train_cluster_id_set = cluster_id_set - train_cluster_id_set
    valid_cluster_id_set = OrderedSet(rnd.sample(list(all_minus_train_cluster_id_set), valid_len))
    test_cluster_id_set = OrderedSet(all_minus_train_cluster_id_set - valid_cluster_id_set)

    train_cluster_dict = {
        cluster_id: cluster_dict[cluster_id] for cluster_id in train_cluster_id_set
    }
    valid_cluster_dict = {
        cluster_id: cluster_dict[cluster_id] for cluster_id in valid_cluster_id_set
    }
    test_cluster_dict = {cluster_id: cluster_dict[cluster_id] for cluster_id in test_cluster_id_set}

    return train_cluster_dict, valid_cluster_dict, test_cluster_dict


def split_clusters(cluster_dict, train_proportion, valid_proportion, random_seed):
    singleton_cluster_dict = {
        cluster_id: cluster for cluster_id, cluster in cluster_dict.items() if len(cluster) == 1
    }
    plural_cluster_dict = {
        cluster_id: cluster for cluster_id, cluster in cluster_dict.items() if len(cluster) > 1
    }
    singleton_len = len(singleton_cluster_dict)
    plural_len = len(plural_cluster_dict)

    train_singleton_len = int(train_proportion * singleton_len)
    valid_singleton_len = int(valid_proportion * singleton_len)

    train_plural_len = int(train_proportion * plural_len)
    valid_plural_len = int(valid_proportion * plural_len)

    (
        train_singleton_cluster_dict,
        valid_singleton_cluster_dict,
        test_singleton_cluster_dict,
    ) = _split_cluster_dict(
        cluster_dict=singleton_cluster_dict,
        train_len=train_singleton_len,
        valid_len=valid_singleton_len,
        random_seed=random_seed,
    )
    (
        train_plural_cluster_dict,
        valid_plural_cluster_dict,
        test_plural_cluster_dict,
    ) = _split_cluster_dict(
        cluster_dict=plural_cluster_dict,
        train_len=train_plural_len,
        valid_len=valid_plural_len,
        random_seed=random_seed,
    )

    logger.info(
        "Singleton cluster sizes (train, valid, test):"
        + str(
            (
                len(train_singleton_cluster_dict),
                len(valid_singleton_cluster_dict),
                len(test_singleton_cluster_dict),
            )
        )
    )
    logger.info(
        "Plural cluster sizes (train, valid, test):"
        + str(
            (
                len(train_plural_cluster_dict),
                len(valid_plural_cluster_dict),
                len(test_plural_cluster_dict),
            )
        )
    )

    train_cluster_dict = {**train_singleton_cluster_dict, **train_plural_cluster_dict}
    valid_cluster_dict = {**valid_singleton_cluster_dict, **valid_plural_cluster_dict}
    test_cluster_dict = {**test_singleton_cluster_dict, **test_plural_cluster_dict}

    return train_cluster_dict, valid_cluster_dict, test_cluster_dict


def _filtered_record_dict_from_cluster_dict(record_dict, cluster_dict):
    return {id_: record_dict[id_] for cluster in cluster_dict.values() for id_ in cluster}


def split_record_dict_on_clusters(
    record_dict, cluster_field, train_proportion, valid_proportion, random_seed
):
    cluster_dict = record_dict_to_cluster_dict(record_dict, cluster_field)
    train_cluster_dict, valid_cluster_dict, test_cluster_dict = split_clusters(
        cluster_dict, train_proportion, valid_proportion, random_seed
    )
    return (
        _filtered_record_dict_from_cluster_dict(record_dict, train_cluster_dict),
        _filtered_record_dict_from_cluster_dict(record_dict, valid_cluster_dict),
        _filtered_record_dict_from_cluster_dict(record_dict, test_cluster_dict),
    )


def split_id_pairs(id_pairs, train_proportion, valid_proportion, random_seed):
    rnd = random.Random(random_seed)

    pairs_len = len(id_pairs)
    train_len = int(train_proportion * pairs_len)
    valid_len = int(valid_proportion * pairs_len)

    id_pairs = OrderedSet(id_pairs)  # ensure deterministic order
    train_pos_pair_set = OrderedSet(rnd.sample(list(id_pairs), train_len))
    all_minus_train_set = id_pairs - train_pos_pair_set
    valid_pos_pair_set = OrderedSet(rnd.sample(list(all_minus_train_set), valid_len))
    test_pos_pair_set = all_minus_train_set - valid_pos_pair_set

    logger.info(
        "Positive pair counts (train, valid, test):"
        + str(
            (
                len(train_pos_pair_set),
                len(valid_pos_pair_set),
                len(test_pos_pair_set),
            )
        )
    )
    return train_pos_pair_set, valid_pos_pair_set, test_pos_pair_set


def split_record_dict_from_id_pairs(
    record_dict,
    train_pos_pair_set,
    valid_pos_pair_set,
    test_pos_pair_set,
    valid_neg_pair_set=None,
    train_neg_pair_set=None,
    test_neg_pair_set=None,
):
    train_record_dict = {
        id_: record_dict[id_]
        for pair in train_pos_pair_set | (train_neg_pair_set or set())
        for id_ in pair
    }
    valid_record_dict = {
        id_: record_dict[id_]
        for pair in valid_pos_pair_set | (valid_neg_pair_set or set())
        for id_ in pair
    }
    test_record_dict = {
        id_: record_dict[id_]
        for pair in test_pos_pair_set | (test_neg_pair_set or set())
        for id_ in pair
    }
    return train_record_dict, valid_record_dict, test_record_dict


def compute_max_str_len(field_val_gen, is_multitoken, tokenizer):
    actual_max_str_len = 0
    for field_val in field_val_gen:
        if not is_multitoken:
            str_len = len(field_val)
        else:
            tokens = tokenizer(field_val)
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


def compute_vocab_counter(field_val_gen, tokenizer):
    vocab_counter = Counter()
    for field_val in field_val_gen:
        tokens = tokenizer(field_val)
        vocab_counter.update(tokens)
    return vocab_counter


def id_pairs_to_cluster_mapping_and_dict(id_pairs, record_dict):
    uf = UnionFind()
    uf.union_pairs(id_pairs)
    cluster_dict = uf.component_dict()

    # restructure cluster_dict to have sequential cluster ids
    cluster_dict = dict(enumerate(cluster_dict.values()))

    # sort to always have smaller id on left of pair tuple
    for c in cluster_dict.values():
        c.sort()

    # now build cluster_mapping using the new cluster ids
    cluster_mapping = {
        id_: cluster_id for cluster_id, cluster in cluster_dict.items() for id_ in cluster
    }

    # add singleton clusters
    current_singleton_cluster_id = max(cluster_dict.keys()) + 1
    for id_ in record_dict.keys() - cluster_mapping.keys():
        cluster_mapping[id_] = current_singleton_cluster_id
        cluster_dict[current_singleton_cluster_id] = [id_]
        current_singleton_cluster_id += 1

    return cluster_mapping, cluster_dict


def assign_clusters(record_dict, cluster_field, cluster_mapping):
    for id_, record in record_dict.items():
        record[cluster_field] = cluster_mapping[id_]


def subdict(d, keys):
    return {k: d[k] for k in keys}


def tensor_dict_to_device(tensor_dict, device):
    return {field: t.to(device) if t is not None else None for field, t in tensor_dict.items()}
