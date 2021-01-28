def precision(found_pair_set, true_pair_set):
    true_positives = found_pair_set & true_pair_set
    false_positives = found_pair_set - true_pair_set
    return len(true_positives) / (len(true_positives) + len(false_positives))


def recall(found_pair_set, true_pair_set):
    true_positives = found_pair_set & true_pair_set
    return len(true_positives) / len(true_pair_set)
