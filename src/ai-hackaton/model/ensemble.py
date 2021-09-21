import warnings

import neptune.new as neptune
import neptune.new.integrations.sklearn as npt_utils
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


def train_kfold_svm(
    n_fold: int,
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
) -> np.ndarray:
    kf = StratifiedKFold(n_splits=n_fold, random_state=42, shuffle=True)
    splits = kf.split(X, y)
    svm_oof = np.zeros((X.shape[0], 61))
    svm_preds = np.zeros((X_test.shape[0], 61))

    run = neptune.init(
        project="ds-wook/ai-hackaton", tags=["Extra-Tree", "Stratified KFold"]
    )

    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        print("Fold :", fold)
        # create dataset
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]
        # model
        model = SVC(C=20, kernel="rbf", gamma="auto", probability=True)
        model.fit(X_train, y_train)
        # validation
        svm_oof[valid_idx] = model.predict_proba(X_valid)
        svm_preds += model.predict_proba(X_test) / n_fold
        print(f"Performance Log Loss: {log_loss(y_valid, svm_oof[valid_idx])}")

        run[f"cls_summary/fold_{fold}"] = npt_utils.create_classifier_summary(
            model, X_train, X_valid, y_train, y_valid
        )

    print(f"Total Performance Log Loss: {log_loss(y, svm_oof)}")

    run.stop()

    return svm_preds
