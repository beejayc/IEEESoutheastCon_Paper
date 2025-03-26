import base64
import os
import numpy as np


def calculate_macro_scores(confusion_matrix):
    """
    Calculates the overall macro precision, recall, and F1 score from a confusion matrix.

    Args:
      confusion_matrix: A 2D numpy array representing the confusion matrix, 
                        where rows are true classes and columns are predictions.

    Returns:
      A tuple containing macro precision, macro recall, and macro F1 score.
    """

    num_classes = confusion_matrix.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)

    for i in range(num_classes):
        true_positives = confusion_matrix[i, i]
        false_positives = np.sum(confusion_matrix[:, i]) - true_positives
        false_negatives = np.sum(confusion_matrix[i, :]) - true_positives

        # Avoid division by zero
        if true_positives + false_positives > 0:
            precision[i] = true_positives / (true_positives + false_positives)
        if true_positives + false_negatives > 0:
            recall[i] = true_positives / (true_positives + false_negatives)

        if precision[i] + recall[i] > 0:
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1_score)

    return macro_precision, macro_recall, macro_f1
