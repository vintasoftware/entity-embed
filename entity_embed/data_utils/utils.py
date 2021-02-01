import itertools
import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


def row_dict_to_id_pairs(row_dict, cluster_attr, out_type=list):
    cluster_id_to_ids = defaultdict(list)
    for id_, row in row_dict.items():
        cluster_id_to_ids[row[cluster_attr]].append(id_)

    return out_type(
        pair
        for cluster in cluster_id_to_ids.values()
        for pair in itertools.combinations(cluster, 2)
    )


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

    # Ensure max_str_len is pair to enable pooling later
    if actual_max_str_len % 2 != 0:
        logger.info(
            f"{actual_max_str_len=} must be pair to enable NN pooling. "
            f"Updating to {actual_max_str_len + 1}"
        )
        actual_max_str_len += 1

    return actual_alphabet, actual_max_str_len
