def pair_entity_ratio(found_pair_set_len, entity_count):
    return found_pair_set_len / entity_count


def precision_and_recall(found_pair_set, true_pair_set):
    true_positives = found_pair_set & true_pair_set
    false_positives = found_pair_set - true_pair_set
    precision = len(true_positives) / (len(true_positives) + len(false_positives))
    recall = len(true_positives) / len(true_pair_set)
    return precision, recall


def f1_score(precision, recall):
    try:
        return (2 * precision * recall) / (precision + recall)
    except ZeroDivisionError:
        return 0
