import warnings
from typing import Callable, Sequence, Union

import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils
import numpy as np
import optuna
import pandas as pd
import yaml
from catboost import CatBoostClassifier, Pool
from hydra.utils import to_absolute_path
from lightgbm import LGBMClassifier
from neptune.new.exceptions import NeptuneMissingApiTokenException
from optuna.integration import LightGBMPruningCallback, XGBoostPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


class BayesianOptimizer:
    def __init__(
        self, objective_function: Callable[[Trial], Union[float, Sequence[float]]]
    ):
        self.objective_function = objective_function

    def build_study(self, trials: FrozenTrial, verbose: bool = False):
        try:
            run = neptune.init(project="ds-wook/ai-hackaton", tags="optimization")

            neptune_callback = optuna_utils.NeptuneCallback(
                run, plots_update_freq=1, log_plot_slice=False, log_plot_contour=False
            )
            sampler = TPESampler(seed=42)
            study = optuna.create_study(
                study_name="TPE Optimization",
                direction="minimize",
                sampler=sampler,
                pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            )
            study.optimize(
                self.objective_function, n_trials=trials, callbacks=[neptune_callback]
            )
            run.stop()

        except NeptuneMissingApiTokenException:
            sampler = TPESampler(seed=42)
            study = optuna.create_study(
                study_name="optimization", direction="minimize", sampler=sampler
            )
            study.optimize(self.objective_function, n_trials=trials)
        if verbose:
            self.display_study_statistics(study)

        return study

    @staticmethod
    def display_study_statistics(study: Study):
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    '{key}': {value},")

    @staticmethod
    def lgbm_save_params(study: Study, params_name: str):
        params = study.best_trial.params
        params["random_state"] = 42
        params["n_estimators"] = 10000
        params["boosting_type"] = "gbdt"
        params["objective"] = "multiclass"
        params["metric"] = "multi_logloss"
        params["n_jobs"] = -1

        with open(to_absolute_path("../../config/train/lgbm.yaml")) as f:
            train_dict = yaml.load(f, Loader=yaml.FullLoader)
        train_dict["params"] = params

        with open(to_absolute_path("../../config/train/" + params_name), "w") as p:
            yaml.dump(train_dict, p)

    @staticmethod
    def xgb_save_params(study: optuna.create_study, params_name: str):
        params = study.best_trial.params
        params["random_state"] = 42
        params["n_estimators"] = 10000
        params["n_jobs"] = -1
        params["objective"] = "multi:softmax"
        params["eval_metric"] = "mlogloss"
        with open(to_absolute_path("../../config/train/xgb.yaml")) as f:
            train_dict = yaml.load(f, Loader=yaml.FullLoader)
        train_dict["params"] = params

        with open(to_absolute_path("../../config/train/" + params_name), "w") as p:
            yaml.dump(train_dict, p)


def lgbm_objective(
    trial: FrozenTrial,
    X: pd.DataFrame,
    y: pd.Series,
    n_fold: int,
) -> float:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-01, 2e-01),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-02, 1),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-02, 1),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-02, 1),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.1, 1),
        "subsample": trial.suggest_uniform("subsample", 0.1, 1),
        "min_child_samples": trial.suggest_int("min_child_samples", 32, 128),
        "max_depth": trial.suggest_int("max_depth", 4, 32),
        "n_estimators": 10000,
        "random_state": 42,
        "objective": "multiclass",
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "n_jobs": -1,
    }
    pruning_callback = LightGBMPruningCallback(
        trial, "multi_logloss", valid_name="valid_1"
    )

    kf = StratifiedKFold(n_splits=n_fold, random_state=42, shuffle=True)
    splits = kf.split(X, y)
    lgbm_oof = np.zeros((X.shape[0], 61))

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        # create dataset
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        # model
        model = LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=50,
            verbose=False,
            callbacks=[pruning_callback],
        )
        # validation
        lgbm_oof[valid_idx] = model.predict_proba(X_valid)

    score = log_loss(y, lgbm_oof)
    return score


def xgb_objective(
    trial: FrozenTrial,
    X: pd.DataFrame,
    y: pd.Series,
    n_fold: int,
) -> Callable[[Trial], float]:
    params = {
        "random_state": 42,
        "n_estimators": 10000,
        "objective": "multi:softmax",
        "eval_metric": "mlogloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 3e-5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 9e-2),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "gamma": trial.suggest_float("gamma", 0.1, 1),
    }

    kf = StratifiedKFold(n_splits=n_fold, random_state=42, shuffle=True)
    splits = kf.split(X, y)
    xgb_oof = np.zeros((X.shape[0], 61))
    pruning_callback = XGBoostPruningCallback(trial, "validation_1-mlogloss")

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        # create dataset
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        # model
        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            early_stopping_rounds=50,
            verbose=False,
            callbacks=[pruning_callback],
        )
        # validation
        xgb_oof[valid_idx] = model.predict_proba(X_valid)

    score = log_loss(y, xgb_oof)
    return score


def cat_objective(
    trial: FrozenTrial,
    X: pd.DataFrame,
    y: pd.Series,
    n_fold: int,
) -> Callable[[Trial], float]:
    params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "od_type": "Iter",
        "od_wait": 500,
        "random_seed": 42,
        "iterations": 26000,
        "learning_rate": trial.suggest_uniform("learning_rate", 1e-3, 1e-2),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-1, 1.0, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "bagging_temperature": trial.suggest_int("bagging_temperature", 1, 10),
        # "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        # "max_bin": trial.suggest_int("max_bin", 200, 500),
    }

    kf = StratifiedKFold(n_splits=n_fold, random_state=42, shuffle=True)
    splits = kf.split(X, y)
    cat_oof = np.zeros((X.shape[0], 61))

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        # create dataset
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        train_data = Pool(data=X_train, label=y_train)
        valid_data = Pool(data=X_valid, label=y_valid)

        # model
        model = CatBoostClassifier(**params)
        model.fit(
            train_data,
            eval_set=valid_data,
            early_stopping_rounds=50,
            use_best_model=True,
            verbose=False,
        )
        # validation
        cat_oof[valid_idx] = model.predict_proba(X_valid)

    log_score = log_loss(y, cat_oof)
    return log_score
