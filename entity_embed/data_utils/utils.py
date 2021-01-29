import itertools
import math
from collections import defaultdict


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
    # positive solution of n for `y of y = (n * (n - 1)) / 2`
    # where y is pair_count and n is row_count
    return int((1 + math.sqrt(1 + 8 * pair_count)) / 2)
