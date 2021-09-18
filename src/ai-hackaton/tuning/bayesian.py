import warnings
from typing import Callable, Sequence, Union

import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils
import numpy as np
import optuna
import pandas as pd
import yaml
from hydra.utils import to_absolute_path
from lightgbm import LGBMClassifier
from neptune.new.exceptions import NeptuneMissingApiTokenException
from optuna.integration import LightGBMPruningCallback
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial, Trial
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


class BayesianOptimizer:
    def __init__(
        self, objective_function: Callable[[Trial], Union[float, Sequence[float]]]
    ):
        self.objective_function = objective_function

    def build_study(self, trials: FrozenTrial, verbose: bool = False):
        try:
            run = neptune.init(
                project="ds-wook/ai-hackaton", tags="optimization"
            )

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
        params["verbosity"] = -1
        params["n_jobs"] = -1

        with open(to_absolute_path("../../config/train/train.yaml")) as f:
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
        "learning_rate": trial.suggest_float("learning_rate", 1e-02, 2e-01),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
        "num_leaves": trial.suggest_int("num_leaves", 16, 64),
        "min_child_weight": trial.suggest_float("min_child_weight", 20, 50),
        "colsample_bytree": trial.suggest_uniform("feature_fraction", 0.1, 1),
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
    pruning_callback = LightGBMPruningCallback(trial, "RMSPE", valid_name="valid_1")

    group_kf = StratifiedKFold(n_splits=n_fold, random_state=42, shuffle=True)
    splits = group_kf.split(X, y)
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
        lgbm_oof[valid_idx] = model.predict(
            X_valid, num_iteration=model.best_iteration_
        )

    score = log_loss(y, lgbm_oof)
    return score
