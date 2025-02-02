import numpy as np
import json


def features_selection(x, header):
    """Select the important features

    Args:
        x: numpy array, containing the data.
        header: numpy array, matching header, containing the corresponding keys.

    Returns:
        x_important: numpy array, containing the data corresponding to important features.
        header_important: matching header, containing the important keys.
    """
    with open("./data_json/important_data.json", "r") as f:
        important_columns = json.load(f)
    # Find the important indices for the given header
    important_indices = [i for i, col in enumerate(header) if col in important_columns]
    x_important = x[:, important_indices]
    header_important = header[important_indices]
    return x_important, header_important


def cleaning(x, header):
    """Removes unwanted values

    Args:
        x:  numpy array, containing the data.
        header: numpy array, matching header containing the corresponding keys.

    Returns:
        x: numpy array, containing the data where the unwanted values are replaced by a nan value.
    """
    with open("./data_json/important_data_clean.json", "r") as file:
        cleaning_rules = json.load(file)

    for column, rules in cleaning_rules.items():
        if column in header:
            col_idx = np.where(header == column)[0][0]

            if "Cleaning" in rules and rules["Cleaning"]:
                clean_values = [
                    int(value)
                    for value in rules["Cleaning"][0].split(",")
                    if value.strip()
                ]

                for i, val in enumerate(x[:, col_idx]):
                    if val in clean_values:
                        x[i, col_idx] = np.nan

    return x


def replace_nan_values(x):
    """Replace Nan values in a given array by the column means.

    Args:
        x: numpy array, dataset.

    Returns:
        x: numpy array, without Nan values.
    """
    for i in range(x.shape[1]):
        column_values = x[:, i]
        column_mean = np.nanmean(column_values)
        x[:, i] = np.nan_to_num(column_values, nan=column_mean)
    return x


def standardize(x):
    """Standardize the original data set.

    Args:
        x: numpy array, dataset.

    Returns:
        x: numpy array, standardized.
    """
    mean_x = np.mean(x, axis=0)
    std_x = np.std(x, axis=0)

    x -= mean_x
    # Prevent dividing by 0
    zero_std_mask = std_x == 0
    std_x[zero_std_mask] = 1
    x = x / std_x
    # Set features with std 0 to 0 because they dont provide value
    x[:, zero_std_mask] = 0
    return x


def create_pearson_correlation(x, y, number_of_samples=100):
    """Creates the pearson correlation for given arrays x and y

    Args:
        x: numpy array, dataset.
        y: numpy array, labels
        number_of_samples: float, the number of samples with which the correlations are calculated

    Returns:
        correlations: numpy array, the pearson correlations for the given data.
    """
    if number_of_samples > x.shape[0]:
        raise Exception(
            "Number of samples can not be bigger than the number of training samples"
        )
    # Create samples to calculate the correlations
    sample_mask = np.random.randint(0, x.shape[0], size=number_of_samples)
    x_sample = x[sample_mask]
    y_sample = y[sample_mask]

    correlations = []
    for i in range(x_sample.shape[1]):
        feature = x_sample[:, i]
        if np.std(feature) == 0:
            correlations.append(
                0
            )  # If the feature has no variance there is no gain through that feature -> correlation 0
        else:
            corr = np.corrcoef(feature, y_sample)[0, 1]
            if np.isnan(corr):
                corr = 0
            correlations.append(corr)

    return np.array(correlations)


def filter_top_correlated_features(x, train_header, correlations, top_percentage=0.25):
    """Filters a percentage of top correlated features for given correlations

    Args:
        x: numpy array, dataset.
        train_header: numpy array, the header of the dataset
        correlations: numpy array, the correlations to filter
        top_percentage: float, the percentage that should be filtered

    Returns:
        filtered_x: numpy array, the filtered dataset
        train_header: numpy array, the filtered header of the dataset
    """
    num_features_to_keep = int(len(correlations) * top_percentage)
    sorted_indices = np.argsort(np.abs(correlations))[::-1]
    top_indices = sorted_indices[:num_features_to_keep]
    filtered_x = x[:, top_indices]
    train_header = train_header[top_indices]
    return filtered_x, train_header


def preprocess_data(y_train, x_train, x_test, train_header, top_percentage):
    """Creates the pearson correlation for given arrays x and y

    Args:
        y_train: numpy array, labels
        x_train: numpy array, dataset
        x_test: numpy array, testset
        train_header: numpy array, the header of the dataset
        top_percentage: float, the percentage that should be filtered

    Returns:
        x_train_top: numpy array, the filtered dataset
        x_test_top: numpy array, the filtered testset
        train_header_top: numpy array, the filtered header of the dataset
    """
    x_train_sel, train_header_sel = features_selection(x_train, train_header)
    x_test_sel, _ = features_selection(x_test, train_header)

    x_train_cleaned = cleaning(x_train_sel, train_header_sel)
    x_test_sel_cleaned = cleaning(x_test_sel, train_header_sel)

    x_train_non_nan = replace_nan_values(x_train_cleaned)
    x_test_non_nan = replace_nan_values(x_test_sel_cleaned)

    x_train_std = standardize(x_train_non_nan)
    x_test_std = standardize(x_test_non_nan)

    correlations = create_pearson_correlation(x_train_std, y_train, 1000)
    x_train_top, train_header_top = filter_top_correlated_features(
        x_train_std, train_header_sel, correlations, top_percentage
    )
    x_test_top, _ = filter_top_correlated_features(
        x_test_std, train_header_sel, correlations, top_percentage
    )
    return x_train_top, x_test_top, train_header_top


def upsample(x, y, seed):
    """Upsamples the minority class in the data so that the resulting dataset
    has a 50/50 class ratio.

    Args:
        x: numpy array, dataset
        y: numpy array, labels
        seed: int, the seed for np.random

    Returns:
        x_balanced: numpy array, the balanced dataset
        y_balanced: numpy array, balanced labels
    """
    np.random.seed(seed)

    # Identify the unique classes and their counts
    classes, counts = np.unique(y, return_counts=True)
    majority_class = classes[np.argmax(counts)]
    minority_class = classes[np.argmin(counts)]

    x_majority = x[y == majority_class]
    x_minority = x[y == minority_class]

    # Calculate the difference needed to balance
    n_majority = x_majority.shape[0]
    n_minority = x_minority.shape[0]
    n_to_add = n_majority - n_minority

    # Randomly sample with replacement from the minority class
    x_minority_upsampled = x_minority[
        np.random.choice(n_minority, n_to_add, replace=True)
    ]
    y_minority_upsampled = np.full(n_to_add, minority_class)

    x_balanced = np.concatenate((x, x_minority_upsampled), axis=0)
    y_balanced = np.concatenate((y, y_minority_upsampled), axis=0)

    return x_balanced, y_balanced
