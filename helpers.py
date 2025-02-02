import csv
import numpy as np
import os
from costs import (
    compute_loss_mse,
    compute_log_loss,
    calculate_accuracy,
    calculate_f1_score,
)
from data_processing import upsample


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    # load headers
    train_header = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", dtype=str, max_rows=1
    )
    test_header = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", dtype=str, max_rows=1
    )

    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    train_header = train_header[1:]
    test_header = test_header[1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids, train_header, test_header


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return 1 / (1 + np.exp(-t))


def calculate_log_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N,)
        tx: shape=(N,D)
        w:  shape=(D,)

    Returns:
        a vector of shape (D,)
    """
    error = sigmoid(tx @ w) - y
    return (tx.T @ error) / y.shape[0]


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y: numpy array of shape=(N,)
        k_fold: int, the K in K-fold, i.e. the fold num
        seed: int, the seed for np.ranomd

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_ls(y, x, k_indices, k, threshhold, lambda_=None):
    from implementations import least_squares, ridge_regression

    """returns the traing and test loss of as well as accuracy and f1 score for ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        threshhold: scalar, the threshold to map the results
        lambda_:    scalar, the weight for the regularization

    Returns:
        traing and test loss of as well as accuracy and f1 score
    """
    test_indices = k_indices[k]
    train_indices = np.delete(k_indices, (k), axis=0).flatten()

    test_data_x = x[test_indices]
    test_data_y = y[test_indices]
    train_data_x = x[train_indices]
    train_data_y = y[train_indices]

    train_data_x_up, train_data_y_up = upsample(train_data_x, train_data_y, 1)

    if lambda_ is not None:
        w_optimal, final_loss = ridge_regression(
            train_data_y_up, train_data_x_up, lambda_
        )
    else:
        w_optimal, final_loss = least_squares(train_data_y_up, train_data_x_up)

    loss_tr = compute_loss_mse(train_data_y_up, train_data_x_up, w_optimal)
    loss_te = compute_loss_mse(test_data_y, test_data_x, w_optimal)

    y_pred_test = test_data_x @ w_optimal
    y_pred_test = map_results(y_pred_test, threshhold)

    accuracy = calculate_accuracy(test_data_y, y_pred_test)
    f1_score = calculate_f1_score(test_data_y, y_pred_test)
    return loss_tr, loss_te, accuracy, f1_score


def cross_validation_gd(
    y,
    x,
    k_indices,
    k,
    initial_w,
    threshhold,
    stochastic=False,
    max_iters=500,
    gamma=0.1,
):
    from implementations import mean_squared_error_gd, mean_squared_error_sgd

    """returns the traing and test loss of as well as accuracy and f1 score for gradient_descent for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        initial_w:  shape=(D,) the initial feature vector
        threshhold: float, the threshold for mapping the results
        stochastic: boolean, indicates if stochastic gradient descent should be used
        max_iters:  int, the max iterations for gradient descent
        gamma:      float, the step size for gradient descent

    Returns:
        traing and test loss of as well as accuracy and f1 score

    """
    test_indices = k_indices[k]
    train_indices = np.delete(k_indices, (k), axis=0).flatten()

    test_data_x = x[test_indices]
    test_data_y = y[test_indices]
    train_data_x = x[train_indices]
    train_data_y = y[train_indices]

    train_data_x_up, train_data_y_up = upsample(train_data_x, train_data_y, 1)

    if stochastic:
        w_optimal, final_loss = mean_squared_error_sgd(
            train_data_y_up, train_data_x_up, initial_w, max_iters, gamma
        )
    else:
        w_optimal, final_loss = mean_squared_error_gd(
            train_data_y_up, train_data_x_up, initial_w, max_iters, gamma
        )

    loss_tr = compute_loss_mse(train_data_y, train_data_x, w_optimal)
    loss_te = compute_loss_mse(test_data_y, test_data_x, w_optimal)

    y_pred_test = test_data_x @ w_optimal
    y_pred_test = map_results(y_pred_test, threshhold)

    accuracy = calculate_accuracy(test_data_y, y_pred_test)
    f1_score = calculate_f1_score(test_data_y, y_pred_test)
    return loss_tr, loss_te, accuracy, f1_score


def cross_validation_log(
    y, x, k_indices, k, initial_w, lambda_=None, max_iters=500, gamma=0.1
):
    from implementations import logistic_regression, reg_logistic_regression

    """return the traing and test loss of as well as accuracy and f1 score of gradient_descent for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          int, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        initial_w:  shape=(D,) the initial feature vector
        threshhold: float, the threshold for mapping the results
        lambda_:    float, the weight for regularization
        max_iters:  int, the max iterations for gradient descent
        gamma:      float, the step size for gradient descent

    Returns:
        traing and test loss of as well as accuracy and f1 score

    """
    test_indices = k_indices[k]
    train_indices = np.delete(k_indices, (k), axis=0).flatten()

    test_data_x = x[test_indices]
    test_data_y = y[test_indices]
    train_data_x = x[train_indices]
    train_data_y = y[train_indices]

    train_data_x_up, train_data_y_up = upsample(train_data_x, train_data_y, 1)

    if lambda_ is not None:
        w_optimal, final_loss = reg_logistic_regression(
            train_data_y_up, train_data_x_up, lambda_, initial_w, max_iters, gamma
        )
    else:
        w_optimal, final_loss = logistic_regression(
            train_data_y_up, train_data_x_up, initial_w, max_iters, gamma
        )

    loss_tr = compute_log_loss(train_data_y, train_data_x, w_optimal)
    loss_te = compute_log_loss(test_data_y, test_data_x, w_optimal)

    y_pred_test = test_data_x @ w_optimal
    y_pred_test = map_results(y_pred_test, 0.5)

    accuracy = calculate_accuracy(test_data_y, y_pred_test)
    f1_score = calculate_f1_score(test_data_y, y_pred_test)
    return loss_tr, loss_te, accuracy, f1_score


def split_data_rand(y, tx, ratio, seed=1):
    """
    Randomly split the dataset, based on the split ratio and the given seed.

    Parameters:
    y: dataset
    tx: dataset
    ratio: ratio of the split
    seed=1

    Returns:
    x_tr: x training set
    y_tr: y training set
    x_te: x test set
    y_te: y test set
    """
    np.random.seed(seed)

    # Generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[:index_split]
    index_te = indices[index_split:]

    # Create splits
    x_tr = tx[index_tr]
    y_tr = y[index_tr]
    x_te = tx[index_te]
    y_te = y[index_te]

    # Return splits
    return x_tr, y_tr, x_te, y_te


def map_results(y_test, threshhold):
    """map labels for given threshold.

    Args:
        y_test: numpy array, labels
        threshhold: float, the threshold for mapping the results

    Returns:
        numpy array, the mapped labels
    """
    return np.where(y_test > threshhold, 1, -1)


def map_y(y):
    """map labels from {-1, 1} to {0,1}.

    Args:
        y: numpy array, labels

    Returns:
        numpy array, only containing 0 and 1
    """
    return np.where(y == 1, 1, 0)


def initialize_w(x_train):
    """initialize w for regression

    Args:
        x_train: numpy array of shape=(N,D), dataset

    Returns:
        numpy array of shape=(D,), the initial feature vector
    """
    w = np.ones(x_train.shape[1]) * 0.01
    return w
