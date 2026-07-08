# Model factory for the comparison study.
# Provides a single ``make_model(name, params)`` entry point returning an object
# with the sklearn ``fit(X, y)`` / ``predict(X)`` interface for every model in
# the comparison: Linear, Polynomial, SVR, RandomForest, GradientBoosting,
# XGBoost, LightGBM, GaussianProcess, ANN, and KAN.
# Torch models (ANN, KAN) are wrapped so they expose the same interface.

from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
#  ANN
# ======================================================================
class DynamicANN(nn.Module):
    # Feed-forward network with a configurable hidden-layer topology.

    def __init__(self, input_dim: int, hidden_layers: List[int],
                 activation: str = "relu", dropout_rate: float = 0.0):
        # Build hidden layers, activation, and optional dropout.
        super().__init__()
        self.activation_name = activation
        act = self._get_activation(activation)
        layers, prev = [], input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), act]
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        # Map an activation name to a torch module.
        return {
            "relu": nn.ReLU(), "leaky_relu": nn.LeakyReLU(0.1), "elu": nn.ELU(),
            "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(), "selu": nn.SELU(), "gelu": nn.GELU(),
        }.get(name, nn.ReLU())

    def forward(self, x):
        # Forward pass through the sequential network.
        return self.network(x)


class _EarlyStopping:
    # Stop training when validation loss stops improving.

    def __init__(self, patience=50, min_delta=1e-5):
        # Track the best validation loss and remaining patience.
        self.patience, self.min_delta = patience, min_delta
        self.counter, self.best_loss, self.best_state, self.stop = 0, float("inf"), None, False

    def __call__(self, loss, model):
        # Update state from the current validation loss; return True when stopping.
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

    def restore(self, model):
        # Reload the best checkpoint into the model.
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class ANNRegressor:
    # sklearn-style wrapper around DynamicANN with early stopping.

    def __init__(self, hidden_layers=(64, 32, 16), activation="relu", learning_rate=0.001,
                 dropout_rate=0.0, epochs=2000, batch_size=16, patience=50, random_state=42):
        # Store ANN hyperparameters for sklearn-style training.
        self.hidden_layers = list(hidden_layers)
        self.activation = activation
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.model = None
        self.input_dim = None
        self.epochs_run = 0

    def get_config(self):
        # Return a JSON-serializable summary of the trained ANN settings.
        return {
            "architecture": str(self.hidden_layers),
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "dropout_rate": self.dropout_rate,
        }

    def fit(self, X, y):
        # Train the ANN with an internal validation split and early stopping.
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float32)
        self.input_dim = X.shape[1]

        # Internal split for early stopping.
        n_val = max(1, int(len(X) * 0.15))
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(len(X))
        tr, va = idx[:-n_val], idx[-n_val:]
        if len(tr) == 0:
            tr, va = idx, idx

        model = DynamicANN(self.input_dim, self.hidden_layers, self.activation, self.dropout_rate).to(DEVICE)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr])),
            batch_size=self.batch_size, shuffle=True,
        )
        Xva = torch.from_numpy(X[va]).to(DEVICE)
        yva = torch.from_numpy(y[va]).to(DEVICE)

        crit = nn.MSELoss()
        opt = optim.Adam(model.parameters(), lr=self.learning_rate)
        stopper = _EarlyStopping(self.patience)

        for epoch in range(self.epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = crit(model(xb).squeeze(), yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vloss = crit(model(Xva).squeeze(), yva).item()
            self.epochs_run = epoch + 1
            if stopper(vloss, model):
                break
        stopper.restore(model)
        self.model = model
        return self

    def predict(self, X):
        # Predict target values for a feature matrix.
        self.model.eval()
        X = torch.from_numpy(np.asarray(X, np.float32)).to(DEVICE)
        with torch.no_grad():
            return self.model(X).squeeze().cpu().numpy().reshape(-1)


class KANRegressor:
    # sklearn-style wrapper around a pykan KAN.

    def __init__(self, width=(16, 1), grid_size=5, spline_order=3, learning_rate=0.01,
                 epochs=200, batch_size=16, patience=30, random_state=42):
        # Store KAN hyperparameters for sklearn-style training.
        self.width = list(width)
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.model = None

    def get_config(self):
        # Return a JSON-serializable summary of the trained KAN settings.
        return {
            "width": str(self.width), "grid_size": self.grid_size,
            "spline_order": self.spline_order, "learning_rate": self.learning_rate,
        }

    def fit(self, X, y):
        # Train the KAN with an internal validation split and early stopping.
        from kan_model import KAN

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float32)
        full_width = [X.shape[1]] + list(self.width)

        n_val = max(1, int(len(X) * 0.15))
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(len(X))
        tr, va = idx[:-n_val], idx[-n_val:]
        if len(tr) == 0:
            tr, va = idx, idx

        model = KAN(full_width, grid_size=self.grid_size, spline_order=self.spline_order).to(DEVICE)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr])),
            batch_size=self.batch_size, shuffle=True,
        )
        Xva = torch.from_numpy(X[va]).to(DEVICE)
        yva = torch.from_numpy(y[va]).to(DEVICE)
        crit = nn.MSELoss()
        opt = optim.Adam(model.parameters(), lr=self.learning_rate)
        stopper = _EarlyStopping(self.patience)

        for epoch in range(self.epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = crit(model(xb).squeeze(), yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vloss = crit(model(Xva).squeeze(), yva).item()
            if stopper(vloss, model):
                break
        stopper.restore(model)
        self.model = model
        return self

    def predict(self, X):
        # Predict target values for a feature matrix.
        self.model.eval()
        X = torch.from_numpy(np.asarray(X, np.float32)).to(DEVICE)
        with torch.no_grad():
            return self.model(X).squeeze().cpu().numpy().reshape(-1)


# ======================================================================
#  Factory
# ======================================================================
def make_model(name: str, params: Dict[str, Any], random_state: int = 42,
               ann_defaults: Dict[str, Any] = None, kan_defaults: Dict[str, Any] = None):
    # Instantiate one regression model from a hyperparameter dict.
    ann_defaults = ann_defaults or {}
    kan_defaults = kan_defaults or {}
    p = dict(params)

    if name == "Linear":
        return LinearRegression()
    if name == "Polynomial":
        degree = p.get("degree", 2)
        alpha = p.get("alpha", 1.0)
        return make_pipeline(PolynomialFeatures(degree, include_bias=False), Ridge(alpha=alpha, random_state=random_state))
    if name == "SVR":
        return SVR(kernel=p.get("kernel", "rbf"), C=p.get("C", 10.0),
                   epsilon=p.get("epsilon", 0.1), gamma=p.get("gamma", "scale"))
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=p.get("n_estimators", 200), max_depth=p.get("max_depth", None),
            min_samples_split=p.get("min_samples_split", 2), random_state=random_state, n_jobs=-1)
    if name == "GradientBoosting":
        return GradientBoostingRegressor(
            n_estimators=p.get("n_estimators", 200), max_depth=p.get("max_depth", 3),
            learning_rate=p.get("learning_rate", 0.05), random_state=random_state)
    if name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=p.get("n_estimators", 200), max_depth=p.get("max_depth", 3),
            learning_rate=p.get("learning_rate", 0.05), random_state=random_state,
            n_jobs=-1, verbosity=0)
    if name == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            n_estimators=p.get("n_estimators", 200), max_depth=p.get("max_depth", -1),
            learning_rate=p.get("learning_rate", 0.05), random_state=random_state,
            n_jobs=-1, verbosity=-1)
    if name == "GaussianProcess":
        kernel = ConstantKernel(1.0) * RBF(length_scale=p.get("length_scale", 1.0)) + WhiteKernel(p.get("noise", 1.0))
        return GaussianProcessRegressor(kernel=kernel, alpha=p.get("alpha", 1e-6),
                                        normalize_y=True, random_state=random_state)
    if name == "ANN":
        return ANNRegressor(
            hidden_layers=p["architecture"], activation=p["activation"],
            learning_rate=p["learning_rate"], dropout_rate=p.get("dropout_rate", 0.0),
            epochs=ann_defaults.get("epochs", 2000), batch_size=ann_defaults.get("batch_size", 16),
            patience=ann_defaults.get("early_stopping_patience", 50), random_state=random_state)
    if name == "KAN":
        return KANRegressor(
            width=p["width"], grid_size=p.get("grid_size", 5), spline_order=p.get("spline_order", 3),
            learning_rate=p.get("learning_rate", 0.01),
            epochs=kan_defaults.get("epochs", 200), batch_size=kan_defaults.get("batch_size", 16),
            patience=kan_defaults.get("early_stopping_patience", 30), random_state=random_state)
    raise ValueError(f"Unknown model: {name}")


def get_param_grids(training_cfg: Dict[str, Any]) -> "dict[str, list]":
    # Build the {model_name: [param dicts]} search space from config.
    from itertools import product

    grids: "dict[str, list]" = {}

    def combos(cfg, keys):
        values = [cfg.get(k, [None]) for k in keys]
        return [dict(zip(keys, c)) for c in product(*values)]

    if "linear" in training_cfg:
        grids["Linear"] = [{}]
    if "polynomial" in training_cfg:
        grids["Polynomial"] = combos(training_cfg["polynomial"], ["degree", "alpha"])
    if "svr" in training_cfg:
        grids["SVR"] = combos(training_cfg["svr"], ["kernel", "C", "epsilon", "gamma"])
    if "random_forest" in training_cfg:
        grids["RandomForest"] = combos(training_cfg["random_forest"], ["n_estimators", "max_depth", "min_samples_split"])
    if "gradient_boosting" in training_cfg:
        grids["GradientBoosting"] = combos(training_cfg["gradient_boosting"], ["n_estimators", "max_depth", "learning_rate"])
    if "xgboost" in training_cfg:
        grids["XGBoost"] = combos(training_cfg["xgboost"], ["n_estimators", "max_depth", "learning_rate"])
    if "lightgbm" in training_cfg:
        grids["LightGBM"] = combos(training_cfg["lightgbm"], ["n_estimators", "max_depth", "learning_rate"])
    if "gaussian_process" in training_cfg:
        grids["GaussianProcess"] = combos(training_cfg["gaussian_process"], ["length_scale", "noise", "alpha"])
    if "ann" in training_cfg:
        ann = training_cfg["ann"]
        grids["ANN"] = [
            {"architecture": a, "activation": act, "learning_rate": lr, "dropout_rate": dr}
            for a in ann.get("architectures", [[64, 32, 16]])
            for act in ann.get("activation_functions", ["relu"])
            for lr in ann.get("learning_rates", [0.001])
            for dr in ann.get("dropout_rates", [0.0])
        ]
    if "kan" in training_cfg:
        kan = training_cfg["kan"]
        grids["KAN"] = [
            {"width": w, "grid_size": g, "spline_order": s, "learning_rate": lr}
            for w in kan.get("widths", [[16, 1]])
            for g in kan.get("grid_size", [5])
            for s in kan.get("spline_order", [3])
            for lr in kan.get("learning_rate", [0.01])
        ]
    return grids
