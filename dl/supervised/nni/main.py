import nni
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import skorch
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring

LOG = logging.getLogger("skorch_classification")


def load_data():
    """Load dataset"""
    data_original = pd.read_csv("creditcard.csv")
    X = data_original.iloc[:, 1:-1]
    y = data_original["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    scaler_std = StandardScaler()
    X_train_copy = X_train.copy(deep=True)
    X_train = scaler_std.fit_transform(X_train_copy)
    X_val = scaler_std.transform(X_val)

    X_train, y_train = X_train.astype(np.float32), y_train.to_numpy().astype(np.int64)
    X_val, y_val = X_val.astype(np.float32), y_val.to_numpy().astype(np.int64)

    return X_train, X_val, y_train, y_val


def get_default_parameters():
    """get default parameters"""
    # params = {"C": 1.0, "kernel": "linear", "degree": 3, "gamma": 0.01, "coef0": 0.01}
    params = {"hidden_units": 20, "dropout": 0.5, "lr": 0.01}
    return params


class AntiFraudNet1(nn.Module):
    def __init__(
        self,
        input_size=29,  # the feature dimension of inputs
        hidden_units=20,  # initialize the number of hidden units,following the rule of thumb: N_h = (N_in + N_out)*2/3
        nonlin=nn.ReLU(),  # initialize the activation function for hidden layer
        dropout=0.5,  # initialize the dropout
        output_size=2,  # initialize the dimension of outputs
    ):
        super(AntiFraudNet1, self).__init__()
        # self.input_size = input_size
        # self.hidden_units = hidden_units

        self.dense = nn.Linear(input_size, hidden_units)  # dense layer
        self.output = nn.Linear(hidden_units, output_size)
        self.softmax = nn.Softmax(
            dim=1
        )  # https://discuss.pytorch.org/t/how-to-choose-dim-0-1-for-softmax-or-logsoftmax/52676/2
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, **kwargs):
        X = self.dense(X)
        X = self.nonlin(X)
        X = self.dropout(X)
        X = self.output(X)
        X = self.softmax(X)
        return X


def get_model(PARAMS):
    """Get model according to parameters"""
    weight = torch.tensor([1, 577.88])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetClassifier(module=AntiFraudNet1)
    # model.module = AntiFraudNet1
    model.module__hidden_units = PARAMS.get("hidden_units")
    model.module__dropout = PARAMS.get("dropout")
    model.criterion = torch.nn.NLLLoss
    model.criterion__weight = weight
    model.optimizer = torch.optim.Adam
    model.lr = PARAMS.get("lr")
    model.max_epochs = 15
    model.train_split = None
    model.verbose = 0
    model.device = device

    return model


def run(X_train, X_val, y_train, y_val, model):
    """Train model and predict result"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    # score = model.score(X_test, y_test)
    score = f1_score(y_val, y_pred, average="binary")
    LOG.debug("score: %s" % score)
    nni.report_final_result(score)


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    X_train, X_val, y_train, y_val = load_data()

    try:
        # get parameters from tuner
        RECEIVED_PARAMS = nni.get_next_parameter()
        LOG.debug(RECEIVED_PARAMS)
        PARAMS = get_default_parameters()
        PARAMS.update(RECEIVED_PARAMS)
        LOG.debug(PARAMS)
        model = get_model(PARAMS)
        run(X_train, X_val, y_train, y_val, model)
    except Exception as exception:
        LOG.exception(exception)
        raise
