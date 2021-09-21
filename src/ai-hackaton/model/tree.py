import warnings
from typing import Tuple

import neptune.new as neptune
import neptune.new.integrations.sklearn as npt_utils
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


def train_kfold_ext(
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    kf = StratifiedKFold(n_splits=n_fold, random_state=42, shuffle=True)
    splits = kf.split(X, y)
    ext_oof = np.zeros((X.shape[0], 61))
    ext_preds = np.zeros((X_test.shape[0], 61))

    run = neptune.init(
        project="ds-wook/ai-hackaton", tags=["Extra-Tree", "Stratified KFold"]
    )

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        print("Fold :", fold)
        # create dataset
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        # model
        model = ExtraTreesClassifier(max_depth=10, max_features=0.95, n_estimators=171)
        model.fit(X_train, y_train)
        # validation
        ext_oof[valid_idx] = model.predict_proba(X_valid)
        ext_preds += model.predict_proba(X_test) / n_fold
        print(f"Performance Log Loss: {log_loss(y_valid, ext_oof[valid_idx])}")

        run[f"cls_summary/fold_{fold}"] = npt_utils.create_classifier_summary(
            model, X_train, X_valid, y_train, y_valid
        )

    print(f"Total Performance Log Loss: {log_loss(y, ext_oof)}")

    run.stop()

    return ext_oof, ext_preds
