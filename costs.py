import numpy as np


def compute_loss_mse(y, tx, w):
    """Calculates the loss using MSE

    Args:
        y: shape=(N,)
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - np.matmul(tx, w)
    return 1 / (2 * y.shape[0]) * (np.matmul(np.transpose(e), e))


def compute_loss_mae(y, tx, w):
    """Calculates the loss using MAE

    Args:
        y: shape=(N,)
        tx: shape=(N,D)
        w: shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return np.mean((np.abs(y - tx @ w)))


def compute_log_loss(y, tx, w):
    from helpers import sigmoid

    """Computes the cost by negative log likelihood.

    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w:  shape=(D,)

    Returns:
        a non-negative loss
    """
    sigmoid_w = sigmoid(tx @ w)
    cost = -np.mean(y * np.log(sigmoid_w) + (1 - y) * np.log(1 - sigmoid_w))
    return cost


def calculate_accuracy(y_true, y_pred):
    """Calculates the accuracy metric

    Args:
        y_true: shape=(N,) true labels
        y_pred: shape=(N,) predicted labels

    Returns:
        float, the accuracy score
    """
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)

    return accuracy


def calculate_f1_score(y_true, y_pred):
    """Calculates the F1 score.

    Args:
        y_true: shape=(N,) true labels
        y_pred: shape=(N,) predicted labels

    Returns:
        float, the F1 score
    """
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    false_positives = np.sum((y_pred == 1) & (y_true != 1))
    false_negatives = np.sum((y_pred != 1) & (y_true == 1))

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )

    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return f1_score
