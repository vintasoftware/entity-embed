def pair_entity_ratio(found_pair_set_len, entity_count):
    return found_pair_set_len / entity_count


def precision_and_recall(found_pair_set, true_pair_set):
    true_positives = found_pair_set & true_pair_set
    false_positives = found_pair_set - true_pair_set
    if true_positives:
        precision = len(true_positives) / (len(true_positives) + len(false_positives))
    else:
        precision = 0.0
    recall = len(true_positives) / len(true_pair_set)
    return precision, recall


def f1_score(precision, recall):
    if precision or recall:
        return (2 * precision * recall) / (precision + recall)
    else:
        return 0.0
