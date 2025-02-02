import numpy as np
from costs import compute_loss_mse, compute_log_loss
from helpers import batch_iter, calculate_log_gradient


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N,), N is the number of samples.
        tx: numpy array of shape=(N,D), D is the number of features.
        w: numpy array of shape=(D,), the vector of model parameters.

    Returns:
        An array of shape (D,) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - np.matmul(tx, w)
    return -(1 / y.shape[0]) * (np.matmul(np.transpose(tx), e))


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N,), N is the number of samples.
        tx: numpy array of shape=(N,D), D is the number of features.
        initial_w: numpy array of shape=(D,). The initial guess (or the initialization) for the model parameters
        max_iters: int, the total number of iterations of GD
        gamma: float, the stepsize

    Returns:
        w: numpy array of shape=(D,). The final w obtained from mean squared error gradient descent
        final_loss: float, the final loss for the calculated w
    """
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):

        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient

    final_loss = compute_loss_mse(y, tx, w)
    return w, final_loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N,), N is the number of samples.
        tx: numpy array of shape=(N,D), D is the number of features.
        initial_w: numpy array of shape=(D,). the initial guess (or the initialization) for the model parameters
        batch_size: int, the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: int, the total number of iterations of SGD
        gamma: float, the stepsize

    Returns:
        w: numpy array of shape=(D,). The final w obtained from mean squared error stochastic gradient descent
        final_loss: float, the final loss for the calculated w
    """

    # Define parameters to store w and loss
    batch_size = 1
    w = initial_w

    for n_iter in range(max_iters):
        gradients = []
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            gradient = compute_gradient(minibatch_y, minibatch_tx, w)
            gradients.append(gradient)

        g = (1 / batch_size) * np.sum(gradients, axis=0)
        w = w - gamma * g
    final_loss = compute_loss_mse(y, tx, w)
    return w, final_loss


def least_squares(y, tx):
    """The least squares solution.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w_optimal: optimal weights, numpy array of shape(D,), D is the number of features.
        final_loss: float, the loss for the calculated w
    """
    a = np.transpose(tx) @ tx
    b = np.transpose(tx) @ y
    w_optimal = np.linalg.solve(a, b)
    final_loss = compute_loss_mse(y, tx, w_optimal)
    return w_optimal, final_loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: float, the weight for regularization

    Returns:
        w_optimal: optimal weights, numpy array of shape(D,), D is the number of features.
        final_loss: float, the loss for the calculated w
    """
    a = np.transpose(tx) @ tx
    a = a + 2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1])
    b = np.transpose(tx) @ y
    w_optimal = np.linalg.solve(a, b)
    final_loss = compute_loss_mse(y, tx, w_optimal)
    return w_optimal, final_loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    The gradient descent algorithm using logistic regression

    Args:
        y: numpy array of shape=(N,), N is the number of samples
        tx: numpy array of shape=(N,D), D is the number of features
        initial_w: numpy array of shape=(D,), the initial guess (or the initialization) for the model parameters
        max_iters: int, the total number of iterations of GD
        gamma: float, the stepsize

    Returns:
        w: shape=(D, 1)
        loss: scalar number
    """
    w = initial_w
    for iter in range(max_iters):

        gradient = calculate_log_gradient(y, tx, w)
        w = w - gamma * gradient

    final_loss = compute_log_loss(y, tx, w)
    return w, final_loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Does penalized gradient descent using logistic regression. Returns w and final loss.

    Args:
        y: numpy array of shape=(N,), N is the number of samples
        tx: numpy array of shape=(N, D), D is the number of features
        lambda_: float, the weight for regularization
        initial_w: numpy array of shape=(D,), the initial guess (or the initialization) for the model parameters
        max_iters: int, the total number of iterations of GD
        gamma: float, the stepsize

    Returns:
        w: numpy array of shape=(D, 1)
        loss: scalar number
    """
    w = initial_w
    for iter in range(max_iters):

        gradient = calculate_log_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient

    final_loss = compute_log_loss(y, tx, w)
    return w, final_loss
