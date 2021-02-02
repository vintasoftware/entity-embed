def pair_entity_ratio(found_pair_set, valid_row_dict):
    return len(found_pair_set) / len(valid_row_dict)


def precision_and_recall(found_pair_set, true_pair_set):
    true_positives = found_pair_set & true_pair_set
    false_positives = found_pair_set - true_pair_set
    precision = len(true_positives) / (len(true_positives) + len(false_positives))
    recall = len(true_positives) / len(true_pair_set)
    return precision, recall
