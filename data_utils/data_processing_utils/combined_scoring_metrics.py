#!/usr/bin/env python3


"""
# Metrics
Inspired by https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4


## Classification

- Precision
- Recall
- F1 score
- Matthews Correlation Coefficient
- Accuracy
- Balanced accuracy
- Log Loss
- ROC AUC
- PR-AUC

## Regression

- Root mean squared error
- Mean absolute error
- R squared
- Adjusted R squared
- Mean squared error
- Median absolute error
- Mean absolute percentage error
- Explained variance score
- Maximum residual error
- Mean Poisson deviance
- Mean Gamma deviance
- Mean pinball loss


## Clustering

- Fowlkes–Mallows index
- Silhouette Score and Coefficient
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Homogeneity and Completeness Measure
- V-measure cluster
- Pair confusion matrix
- Contingency matrix
- Completeness score
- Rand index
- Adjusted Rand score
- Mutual info score

## Unsupervised Models

- Rand index
- Mutual information

## Others

- CV error
- Heurtistic methods to find K
- BLEU score (NLP)

"""


def classification_scores(y_true, y_pred, y_prob_scores=None, num_dp=4, messages=True):
    """
    Calculate all relevant classification metrics for a given set of true and predicted labels.

    Parameters
    ----------
    y_true : array-like
        The true labels
    y_pred : array-like
        The predicted labels
    y_prob_scores : array-like, optional
        The predicted probabilities
    num_dp : int, optional
        The number of decimal places to round the scores to, by default 4
    messages : bool, optional
        Whether to print out the classification report, by default True

    Returns
    -------
    dict
        A dictionary of all the classification metrics, with the metric name as the key and the score as the value
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
        hamming_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
        confusion_matrix,
        class_likelihood_ratios,
        cohen_kappa_score,
        brier_score_loss,
        zero_one_loss,
        jaccard_score,
        classification_report,
        ndcg_score,
    )

    score_obj = {}

    # From https://stackoverflow.com/a/50671617
    conf_matrix = confusion_matrix(y_true, y_pred)
    false_pos = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    false_neg = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    true_pos = np.diag(conf_matrix)
    true_neg = conf_matrix.sum() - (false_pos + false_neg + true_pos)

    false_pos = false_pos.astype(float)
    false_neg = false_neg.astype(float)
    true_pos = true_pos.astype(float)
    true_neg = true_neg.astype(float)

    score_obj["matthews_corrcoef"] = round(matthews_corrcoef(y_true, y_pred), num_dp)
    score_obj["zero_one_loss"] = round(zero_one_loss(y_true, y_pred), num_dp)
    score_obj["f1"] = round(f1_score(y_true, y_pred), num_dp)
    score_obj["roc_auc"] = round(roc_auc_score(y_true, y_pred), num_dp)
    score_obj["youden_j"] = round((((true_pos * true_neg) - (false_pos * false_neg)) / ((true_pos + false_neg) + (true_neg + false_pos))), num_dp)
    score_obj["recall_or_sensitivity_or_true_pos_rate"] = round(recall_score(y_true, y_pred), num_dp)
    score_obj["specificity_or_true_neg_rate"] = round((true_neg / (true_neg + false_pos)), num_dp)
    score_obj["precision_or_ppv"] = round(precision_score(y_true, y_pred), num_dp)
    score_obj["false_discovery_rate"] = round((false_pos / (true_pos + false_pos)), num_dp)
    score_obj["neg_pred_value"] = round((true_neg / (true_neg + false_neg)), num_dp)
    score_obj["false_omission_rate"] = round((false_neg / (true_neg + false_neg)), num_dp)
    score_obj["false_pos_rate"] = round((false_pos / (false_pos + true_neg)), num_dp)
    score_obj["false_neg_rate"] = round((false_neg / (true_pos + false_neg)), num_dp)
    score_obj["lr_pos"] = round(class_likelihood_ratios(y_true, y_pred)[0], num_dp)
    score_obj["lr_neg"] = round(class_likelihood_ratios(y_true, y_pred)[1], num_dp)
    score_obj["hamming_loss"] = round(hamming_loss(y_true, y_pred), num_dp)
    score_obj["log_loss"] = round(log_loss(y_true, y_pred), num_dp)
    score_obj["accuracy"] = round(accuracy_score(y_true, y_pred), num_dp)
    score_obj["balanced_accuracy"] = round(balanced_accuracy_score(y_true, y_pred), num_dp)
    score_obj["jaccard_index"] = round(jaccard_score(y_true, y_pred), num_dp)
    score_obj["diag_odds_ratio"] = round(((true_pos * true_neg) / (false_pos * false_neg)), num_dp)

    if y_prob_scores is not None:
        score_obj["cohen_kappa"] = round(cohen_kappa_score(y_true, y_pred), num_dp)
        score_obj["brier_score_loss"] = round(brier_score_loss(y_true, y_prob_scores), num_dp)
        score_obj["ndcg_score"] = round(ndcg_score(y_true, y_prob_scores), num_dp)

    if messages:
        print(classification_report(y_true, y_pred))

    return score_obj
