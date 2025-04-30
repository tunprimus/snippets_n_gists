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

    # mcc = round(matthews_corrcoef(y_true, y_pred), num_dp)
    # zero_one_loss = round(zero_one_loss(y_true, y_pred), num_dp)
    # f1 = round(f1_score(y_true, y_pred), num_dp)
    # roc_auc = round(roc_auc_score(y_true, y_pred), num_dp)
    # youden_j = round((((true_pos * true_neg) - (false_pos * false_neg)) / ((true_pos + false_neg) + (true_neg + false_pos))), num_dp)
    # recall_or_sensitivity_or_true_pos_rate = round(recall_score(y_true, y_pred), num_dp)
    # specificity_or_true_neg_rate = round((true_neg / (true_neg + false_pos)), num_dp)
    # precision_or_ppv = round(precision_score(y_true, y_pred), num_dp)
    # false_discovery_rate = round((false_pos / (true_pos + false_pos)), num_dp)
    # neg_pred_value = round((true_neg / (true_neg + false_neg)), num_dp)
    # false_omission_rate = round((false_neg / (true_neg + false_neg)), num_dp)
    # false_pos_rate = round((false_pos / (false_pos + true_neg)), num_dp)
    # false_neg_rate = round((false_neg / (true_pos + false_neg)), num_dp)
    # lr_pos = round(class_likelihood_ratios(y_true, y_pred)[0], num_dp)
    # lr_neg = round(class_likelihood_ratios(y_true, y_pred)[1], num_dp)
    # hamming_loss = round(hamming_loss(y_true, y_pred), num_dp)
    # log_loss = round(log_loss(y_true, y_pred), num_dp)
    # accuracy = round(accuracy_score(y_true, y_pred), num_dp)
    # balanced_accuracy = round(balanced_accuracy_score(y_true, y_pred), num_dp)
    # jaccard_index = round(jaccard_score(y_true, y_pred), num_dp)
    # diag_odds_ratio = round(((true_pos * true_neg) / (false_pos * false_neg)), num_dp)


    mcc = np.round(matthews_corrcoef(y_true, y_pred), decimals=num_dp)
    zero_one_loss = np.round(zero_one_loss(y_true, y_pred), decimals=num_dp)
    f1 = np.round(f1_score(y_true, y_pred), decimals=num_dp)
    roc_auc = np.round(roc_auc_score(y_true, y_pred), decimals=num_dp)
    youden_j = np.round((((true_pos * true_neg) - (false_pos * false_neg)) / ((true_pos + false_neg) + (true_neg + false_pos))), decimals=num_dp)
    recall_or_sensitivity_or_true_pos_rate = np.round(recall_score(y_true, y_pred), decimals=num_dp)
    specificity_or_true_neg_rate = np.round((true_neg / (true_neg + false_pos)), decimals=num_dp)
    precision_or_ppv = np.round(precision_score(y_true, y_pred), decimals=num_dp)
    false_discovery_rate = np.round((false_pos / (true_pos + false_pos)), decimals=num_dp)
    neg_pred_value = np.round((true_neg / (true_neg + false_neg)), decimals=num_dp)
    false_omission_rate = np.round((false_neg / (true_neg + false_neg)), decimals=num_dp)
    false_pos_rate = np.round((false_pos / (false_pos + true_neg)), decimals=num_dp)
    false_neg_rate = np.round((false_neg / (true_pos + false_neg)), decimals=num_dp)
    lr_pos = np.round(class_likelihood_ratios(y_true, y_pred)[0], decimals=num_dp)
    lr_neg = np.round(class_likelihood_ratios(y_true, y_pred)[1], decimals=num_dp)
    hamming_loss = np.round(hamming_loss(y_true, y_pred), decimals=num_dp)
    log_loss = np.round(log_loss(y_true, y_pred), decimals=num_dp)
    accuracy = np.round(accuracy_score(y_true, y_pred), decimals=num_dp)
    balanced_accuracy = np.round(balanced_accuracy_score(y_true, y_pred), decimals=num_dp)
    jaccard_index = np.round(jaccard_score(y_true, y_pred), decimals=num_dp)
    diag_odds_ratio = np.round(((true_pos * true_neg) / (false_pos * false_neg)), decimals=num_dp)


    score_obj["matthews_corrcoef"] = mcc
    score_obj["zero_one_loss"] = zero_one_loss
    score_obj["f1"] = f1
    score_obj["roc_auc"] = roc_auc
    score_obj["youden_j"] = youden_j[-1]
    score_obj["recall_or_sensitivity_or_true_pos_rate"] = recall_or_sensitivity_or_true_pos_rate
    score_obj["specificity_or_true_neg_rate"] = specificity_or_true_neg_rate[-1]
    score_obj["precision_or_ppv"] = precision_or_ppv
    score_obj["false_discovery_rate"] = false_discovery_rate[-1]
    score_obj["neg_pred_value"] = neg_pred_value[-1]
    score_obj["false_omission_rate"] = false_omission_rate[-1]
    score_obj["false_pos_rate"] = false_pos_rate[-1]
    score_obj["false_neg_rate"] = false_neg_rate[-1]
    score_obj["lr_pos"] = lr_pos
    score_obj["lr_neg"] = lr_neg
    score_obj["hamming_loss"] = hamming_loss
    score_obj["log_loss"] = log_loss
    score_obj["accuracy"] = accuracy
    score_obj["balanced_accuracy"] = balanced_accuracy
    score_obj["jaccard_index"] = jaccard_index
    score_obj["diag_odds_ratio"] = diag_odds_ratio[-1]

    if y_prob_scores is not None:
        cohen_kappa = round(cohen_kappa_score(y_true, y_pred), num_dp)
        brier_score_loss = round(brier_score_loss(y_true, y_prob_scores), num_dp)
        ndcg_score = round(ndcg_score(y_true, y_prob_scores), num_dp)

        score_obj["cohen_kappa"] = cohen_kappa
        score_obj["brier_score_loss"] = brier_score_loss
        score_obj["ndcg_score"] = ndcg_score

    if messages:
        print(classification_report(y_true, y_pred))

    return score_obj
