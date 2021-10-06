import warnings
from typing import Any, Dict, Optional, Tuple, Union

import neptune.new as neptune
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from neptune.new.integrations import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def train_kfold_logistic(
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    verbose: Union[int, bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:

    kf = StratifiedKFold(n_splits=n_fold, random_state=42, shuffle=True)
    splits = kf.split(X, y)
    lr_oof = np.zeros((X.shape[0], 61))
    lr_preds = np.zeros((X_test.shape[0], 61))

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        print("Fold :", fold)
        # create dataset
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        # model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        # validation
        lr_oof[valid_idx] = model.predict_proba(X_valid)
        lr_preds += model.predict_proba(X_test) / n_fold
        print(f"Fold{fold} Performance LogLoss: {log_loss(y_valid, lr_oof[valid_idx])}")
    print(f"Total Performance LogLoss: {log_loss(y, lr_oof)}")

    return lr_oof, lr_preds


def train_kfold_xgb(
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    verbose: Union[int, bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:

    kf = StratifiedKFold(n_splits=n_fold, random_state=42, shuffle=True)
    splits = kf.split(X, y)
    xgb_oof = np.zeros((X.shape[0], 61))
    xgb_preds = np.zeros((X_test.shape[0], 61))

    run = neptune.init(
        project="ds-wook/ai-hackaton", tags=["LightGBM", "Stratified KFold"]
    )

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        print("Fold :", fold)
        neptune_callback = xgboost.NeptuneCallback(
            run=run,
            base_namespace=f"fold_{fold}",
            log_tree=[0, 1, 2, 3],
            max_num_features=10,
        )
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
            callbacks=[neptune_callback],
            verbose=verbose,
        )
        # validation
        xgb_oof[valid_idx] = model.predict_proba(X_valid)
        xgb_preds += model.predict_proba(X_test) / n_fold

    print(f"Total Performance Log Loss: {log_loss(y, xgb_oof)}")
    run.stop()

    return xgb_oof, xgb_preds


def train_kfold_cat(
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    verbose: Union[int, bool] = False,
) -> Tuple[np.ndarray, np.ndarray]:
    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
    splits = folds.split(X, y)
    cb_oof = np.zeros((X.shape[0], 61))
    cb_preds = np.zeros((X_test.shape[0], 61))

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        if verbose:
            print(f"\tFold {fold}\n")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        train_data = Pool(data=X_train, label=y_train)
        valid_data = Pool(data=X_valid, label=y_valid)

        model = CatBoostClassifier(**params)

        model.fit(
            train_data,
            eval_set=valid_data,
            early_stopping_rounds=50,
            use_best_model=True,
            verbose=verbose,
        )

        cb_oof[valid_idx] = model.predict_proba(X_valid)
        cb_preds += model.predict_proba(X_test) / n_fold

    log_score = log_loss(y, cb_oof)
    print(f"Log Loss Score: {log_score:.5f}\n")
    return cb_oof, cb_preds


def train_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    verbose: Union[int, bool] = False,
) -> Tuple[Any]:
    x_train, x_valid, y_train, y_valid = train_test_split(
        X, y, random_state=42, test_size=0.2
    )
    oof_pred = np.zeros((x_valid.shape[0], 61))
    model = XGBClassifier(**params)

    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_valid, y_valid)],
        early_stopping_rounds=100,
        verbose=verbose,
    )

    oof_pred = model.predict_proba(x_valid)
    score = log_loss(y_valid, oof_pred)
    print(f"Score: {score}")
    xgb_pred = model.predict_proba(X_test)
    return xgb_pred


def train_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    verbose: Union[int, bool] = False,
) -> Tuple[Any]:
    model = LGBMClassifier(**params)
    model.fit(X, y, verbose=verbose)
    lgbm_pred = model.predict_proba(X_test)
    return lgbm_pred


def train_cat(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
    verbose: Union[int, bool] = False,
) -> Tuple[Any]:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, random_state=42, test_size=0.2
    )
    oof_pred = np.zeros((X_valid.shape[0], 61))

    train_data = Pool(data=X_train, label=y_train)
    valid_data = Pool(data=X_valid, label=y_valid)
    model = CatBoostClassifier(**params)
    model.fit(
        train_data,
        eval_set=valid_data,
        early_stopping_rounds=50,
        use_best_model=True,
        verbose=verbose,
    )

    oof_pred = model.predict_proba(X_valid)
    score = log_loss(y_valid, oof_pred)
    print(f"Score: {score}")
    cat_pred = model.predict_proba(X_test)
    return cat_pred
