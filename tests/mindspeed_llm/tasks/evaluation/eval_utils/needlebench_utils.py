# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.


def trim_prediction(prediction, reference):
    l08 = int(0.8 * len(reference))
    l12 = int(1.2 * len(reference))
    trimmed_prediction = prediction[:l12]

    if len(trimmed_prediction) > l08 and \
            reference[-1] in trimmed_prediction[l08:]:
        end_pos = l08 + trimmed_prediction[l08:].index(reference[-1]) + 1
        trimmed_prediction = trimmed_prediction[:end_pos]

    return trimmed_prediction


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]