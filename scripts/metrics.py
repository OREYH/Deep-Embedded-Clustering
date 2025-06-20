"""Metrics utilities for clustering models using PyTorch."""

import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment


def sensitivity(y_true, y_pred):
    """Compute sensitivity."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_pos = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_pos = np.sum(np.round(np.clip(y_true, 0, 1)))
    return true_pos / (possible_pos + np.finfo(float).eps)


def specificity(y_true, y_pred):
    """Compute specificity."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_neg = np.sum(np.round(np.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_neg = np.sum(np.round(np.clip(1 - y_true, 0, 1)))
    return true_neg / (possible_neg + np.finfo(float).eps)


def calculate_metrics(loss, y, y_pred):
    """Calculate standard clustering metrics."""
    acc = np.round(accuracy(y_true=y, y_pred=y_pred), 5)
    nmi = np.round(normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(adjusted_rand_score(y, y_pred), 5)
    loss = np.round(loss, 5)
    return acc, ari, loss, nmi


def accuracy(y_true, y_pred):
    """Clustering accuracy using the Hungarian algorithm."""
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum(w[i, j] for i, j in zip(*ind)) * 1.0 / y_pred.size
