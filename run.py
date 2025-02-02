import implementations as impl
import json
from data_processing import preprocess_data, upsample
from helpers import (
    load_csv_data,
    create_csv_submission,
    split_data_rand,
    map_results,
    initialize_w,
    build_k_indices,
    cross_validation_ls,
    cross_validation_gd,
    cross_validation_log,
    initialize_w,
)
from costs import calculate_f1_score

# Load data and create local test set
(
    x_train,
    x_test,
    y_train,
    train_ids,
    test_ids,
    train_header,
    test_header,
) = load_csv_data("./data")
x_train_local, y_train_local, x_test_local, y_test_local = split_data_rand(
    y_train, x_train, 0.9, 1
)
print("loaded data")

# Find best data ratio and threshold for linear regression
ratios = [0.25, 0.5, 0.75, 1]
threshholds = [-0.5, -0.25, 0, 0.25, 0.5, 1]
best_f1 = 0
best_ratio = 0
best_threshhold = 0
balanced = False
data_dict = {}

for ratio in ratios:
    for threshhold in threshholds:
        # Preprocess the data for each ratio
        x_processed_ub, x_test_local_processed, _ = preprocess_data(
            y_train_local, x_train_local, x_test_local, train_header, ratio
        )

        # Test with and without upsampling
        x_processed_b, y_train_local_b = upsample(x_processed_ub, y_train_local, 1)

        # We use ridge regression because of the regularization and the performance
        w_unbalanced, _ = impl.ridge_regression(y_train_local, x_processed_ub, 0.01)
        w_balanced, _ = impl.ridge_regression(y_train_local_b, x_processed_b, 0.01)

        # Create prediction and map it for every threshold
        y_pred_unbalanced = x_test_local_processed @ w_unbalanced
        y_pred_balanced = x_test_local_processed @ w_balanced
        y_pred_unbalanced = map_results(y_pred_unbalanced, threshhold)
        y_pred_balanced = map_results(y_pred_balanced, threshhold)

        # Calculate and save f1 score for each ratio and threshold
        f1_unbalanced = calculate_f1_score(y_test_local, y_pred_unbalanced)
        f1_balanced = calculate_f1_score(y_test_local, y_pred_balanced)
        data_dict[f"{ratio},{threshhold},unbalanced"] = f1_unbalanced
        data_dict[f"{ratio},{threshhold},balanced"] = f1_balanced
        if f1_unbalanced > best_f1:
            best_f1 = f1_unbalanced
            balanced = False
            best_ratio = ratio
            best_threshhold = threshhold

        if f1_balanced > best_f1:
            best_f1 = f1_balanced
            balanced = True
            best_ratio = ratio
            best_threshhold = threshhold

print(
    f"Best ratio {best_ratio}, threshold {best_threshhold}, balanced {balanced}, best f1 {best_f1}"
)
with open("./data_json/ratio_threshold.json", "w") as file:
    json.dump(data_dict, file)


# Find the best model according to f1 score with cross validation

# Initialize training data for all models
x_tr, _, _ = preprocess_data(y_train, x_train, x_test, train_header, best_ratio)

# Setup variables for cross validation
k_fold = 4
indices = build_k_indices(y_train, k_fold, 1)
gammas = [0.01, 0.05, 0.1, 0.2]
lambdas = [10**-3, 10**-2, 10**-1, 10**0, 10**1]
initial_w = initialize_w(x_tr)
best_model = None
best_f1_overall = 0
model_gamma = None
model_lambda = None

# Gradient descent
gd_dict = {}
for gamma in gammas:
    tr_loss_gd = te_loss_gd = acc_gd = f1_gd = 0
    for k in range(k_fold):
        tr_loss, te_loss, acc, f1 = cross_validation_gd(
            y_train, x_tr, indices, k, initial_w, best_threshhold, False, 500, gamma
        )
        tr_loss_gd += tr_loss
        te_loss_gd += te_loss
        acc_gd += acc
        f1_gd += f1
    tr_loss_gd /= k_fold
    te_loss_gd /= k_fold
    acc_gd /= k_fold
    f1_gd /= k_fold
    print(
        f"GD {gamma}: tr loss {tr_loss_gd}, te loss {te_loss_gd}, acc {acc_gd}, f1 {f1_gd}"
    )
    gd_dict[gamma] = (tr_loss_gd, te_loss_gd, acc_gd, f1_gd)
    if f1_gd > best_f1_overall:
        best_model = "Gradient descent"
        model_gamma = gamma
        model_lambda = None
        best_f1_overall = f1_gd
# Save generated data
with open("./data_json/gd_performance.json", "w") as file:
    json.dump(gd_dict, file)

# Stochastic gradient descent
sgd_dict = {}
for gamma in gammas:
    tr_loss_sgd = te_loss_sgd = acc_sgd = f1_sgd = 0
    for k in range(k_fold):
        tr_loss, te_loss, acc, f1 = cross_validation_gd(
            y_train, x_tr, indices, k, initial_w, best_threshhold, True, 500, gamma
        )
        tr_loss_sgd += tr_loss
        te_loss_sgd += te_loss
        acc_sgd += acc
        f1_sgd += f1
    tr_loss_sgd /= k_fold
    te_loss_sgd /= k_fold
    acc_sgd /= k_fold
    f1_sgd /= k_fold
    print(
        f"SGD {gamma}: tr loss {tr_loss_sgd}, te loss {te_loss_sgd}, acc {acc_sgd}, f1 {f1_sgd}"
    )
    sgd_dict[gamma] = (tr_loss_sgd, te_loss_sgd, acc_sgd, f1_sgd)
if f1_sgd > best_f1_overall:
    best_model = "Stochastic gradient descent"
    model_gamma = gamma
    model_lambda = None
    best_f1_overall = f1_sgd
# Save generated data
with open("./data_json/sgd_performance.json", "w") as file:
    json.dump(sgd_dict, file)

# Least squares
ls_dict = {}
tr_loss_ls = te_loss_ls = acc_ls = f1_ls = 0
for k in range(k_fold):
    tr_loss, te_loss, acc, f1 = cross_validation_ls(
        y_train, x_tr, indices, k, best_threshhold, lambda_=None
    )
    tr_loss_ls += tr_loss
    te_loss_ls += te_loss
    acc_ls += acc
    f1_ls += f1
tr_loss_ls /= k_fold
te_loss_ls /= k_fold
acc_ls /= k_fold
f1_ls /= k_fold
print(
    f"LS {gamma}: tr loss {tr_loss_ls}, te loss {te_loss_ls}, acc {acc_ls}, f1 {f1_ls}"
)
ls_dict["ls"] = (tr_loss_ls, te_loss_ls, acc_ls, f1_ls)
if f1_ls > best_f1_overall:
    best_model = "Least squares"
    model_gamma = None
    model_lambda = None
    best_f1_overall = f1_ls
# Save generated data
with open("./data_json/ls_performance.json", "w") as file:
    json.dump(ls_dict, file)

# Ridge regression
ridge_dict = {}
for lambda_ in lambdas:
    tr_loss_ridge = te_loss_ridge = acc_ridge = f1_ridge = 0
    for k in range(k_fold):
        tr_loss, te_loss, acc, f1 = cross_validation_ls(
            y_train, x_tr, indices, k, best_threshhold, lambda_=lambda_
        )
        tr_loss_ridge += tr_loss
        te_loss_ridge += te_loss
        acc_ridge += acc
        f1_ridge += f1
    tr_loss_ridge /= k_fold
    te_loss_ridge /= k_fold
    acc_ridge /= k_fold
    f1_ridge /= k_fold
    print(
        f"Ridge {gamma}: tr loss {tr_loss_ridge}, te loss {te_loss_ridge}, acc {acc_ridge}, f1 {f1_ridge}"
    )
    ridge_dict[lambda_] = (tr_loss_ridge, te_loss_ridge, acc_ridge, f1_ridge)
    if f1_ridge > best_f1_overall:
        best_model = "Ridge regression"
        model_gamma = None
        model_lambda = lambda_
        best_f1_overall = f1_ridge
# Save generated data
with open("./data_json/ridge_performance.json", "w") as file:
    json.dump(ridge_dict, file)

# Logistic regression
log_dict = {}
for gamma in gammas:
    tr_loss_log = te_loss_log = acc_log = f1_log = 0
    for k in range(k_fold):
        tr_loss, te_loss, acc, f1 = cross_validation_log(
            y_train, x_tr, indices, k, initial_w, None, 500, gamma
        )
        tr_loss_log += tr_loss
        te_loss_log += te_loss
        acc_log += acc
        f1_log += f1
    tr_loss_log /= k_fold
    te_loss_log /= k_fold
    acc_log /= k_fold
    f1_log /= k_fold
    print(
        f"Log {gamma}: tr loss {tr_loss_log}, te loss {te_loss_log}, acc {acc_log}, f1 {f1_log}"
    )
    log_dict[gamma] = (tr_loss_log, te_loss_log, acc_log, f1_log)
    if f1_log > best_f1_overall:
        best_model = "Logistic regression"
        model_gamma = gamma
        model_lambda = None
        best_f1_overall = f1_log
# Save generated data
with open("./data_json/log_performance.json", "w") as file:
    json.dump(log_dict, file)

# Regularized logistic regression
reg_log_dict = {}
for gamma in gammas:
    for lambda_ in lambdas:
        tr_loss_reg_log = te_loss_reg_log = acc_reg_log = f1_reg_log = 0
        for k in range(k_fold):
            tr_loss, te_loss, acc, f1 = cross_validation_log(
                y_train, x_tr, indices, k, initial_w, lambda_, 500, gamma
            )
            tr_loss_reg_log += tr_loss
            te_loss_reg_log += te_loss
            acc_reg_log += acc
            f1_reg_log += f1
        tr_loss_reg_log /= k_fold
        te_loss_reg_log /= k_fold
        acc_reg_log /= k_fold
        f1_reg_log /= k_fold
        print(
            f"Log {gamma}, {lambda_}: tr loss {tr_loss_reg_log}, te loss {te_loss_reg_log}, acc {acc_reg_log}, f1 {f1_reg_log}"
        )
        reg_log_dict[f"{gamma},{lambda_}"] = (
            tr_loss_reg_log,
            te_loss_reg_log,
            acc_reg_log,
            f1_reg_log,
        )
        if f1_reg_log > best_f1_overall:
            best_model = "Regularized logistic regression"
            model_gamma = gamma
            model_lambda = lambda_
            best_f1_overall = f1_reg_log
# Save generated data
with open("./data_json/reg_log_performance.json", "w") as file:
    json.dump(reg_log_dict, file)

# Best model with parameters
print(
    f"Best model: {best_model} with gamma {model_gamma}, lambda {model_lambda} and f1 score of {best_f1_overall}"
)

# Create submission for best model
x_train_sub, x_test_sub, _ = preprocess_data(
    y_train, x_train, x_test, train_header, best_ratio
)
x_train_sub, y_train_sub = upsample(x_train_sub, y_train, 1)

w_sub_optimal, loss = impl.mean_squared_error_gd(
    y_train_sub, x_train_sub, initial_w, 500, 0.1
)
y_prediction_sub = x_test_sub @ w_sub_optimal
y_prediction_sub = map_results(y_prediction_sub, best_threshhold)
create_csv_submission(test_ids, y_prediction_sub, "./data/final-submission.csv")
