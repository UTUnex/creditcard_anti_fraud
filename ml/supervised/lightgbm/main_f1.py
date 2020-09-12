import nni
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import logging
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

LOG = logging.getLogger("lightgbm_classification")


def load_data():
    """Load dataset"""
    data_original = pd.read_csv("creditcard.csv")
    X = data_original.iloc[:, 1:-1]
    y = data_original["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    # )

    scaler_std = StandardScaler()
    X_train_copy = X_train.copy(deep=True)
    X_train = scaler_std.fit_transform(X_train_copy)
    # X_val = scaler_std.transform(X_val)

    X_train, y_train = X_train.astype(np.float32), y_train.to_numpy().astype(np.int64)
    # X_val, y_val = X_val.astype(np.float32), y_val.to_numpy().astype(np.int64)

    # return X_train, X_val, y_train, y_val
    return X_train, y_train


def get_default_parameters():
    """get default parameters which will be tuned in the following section"""
    params = {"num_leaves": 31, "max_depth": -1, "min_child_weight": 0.001, "subsample": 1, "colsample_bytree": 1}
    return params


def get_model(PARAMS):
    """Get model according to parameters"""
    scale_pos_weight = 577.88  # scale_pos_weight = number of negative samples / number of positive samples
    model = LGBMClassifier()
    model.num_leaves = PARAMS.get("num_leaves")
    model.max_depth = PARAMS.get("max_depth")
    model.n_estimators = 10000
    model.early_stopping_rounds = 20
    model.scale_pos_weight = scale_pos_weight  # we set this parameter to solve the class imbalance problem
    model.objective = "binary"
    model.min_child_weight = PARAMS.get("min_child_weight")
    model.subsample = PARAMS.get("subsample")
    model.subsample_freq = 1
    model.colsample_bytree = PARAMS.get("colsample_bytree")
    model.random_state = 42
    model.n_jobs = -1
    model.max_bin = 63
    model.device = "gpu"
    model.gpu_use_dp = False
    model.gpu_platform_id = 0
    model.gpu_device_id = 0

    return model


def run(X_train, y_train, model):
    """Train model and predict result"""
    """For conventional machine learning models (not deep learning models), we prefer using cross validation for model selection"""
    str_kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # https://stackoverflow.com/questions/50329349/cross-validation-in-sklearn-do-i-need-to-call-fit-as-well-as-cross-val-score
    score = cross_val_score(model, X_train, y_train, cv=str_kfold, scoring="f1", n_jobs=-1).mean()

    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_val)
    # score = model.score(X_test, y_test)
    # score = f1_score(y_val, y_pred, average="binary")
    LOG.debug("score: %s" % score)
    nni.report_final_result(score)


if __name__ == "__main__":
    np.random.seed(42)
    X_train, y_train = load_data()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        model = get_model(PARAMS)
        run(X_train, y_train, model)
    except Exception as exception:
        LOG.exception(exception)
        raise
