import gc
import warnings
from typing import Any, Callable, Dict, NamedTuple, Optional, Union

import neptune.new as neptune
import numpy as np
import pandas as pd
from neptune.new.integrations.xgboost import NeptuneCallback
from sklearn.model_selection import StratifiedKFold
from utils.utils import LoggerFactory
from xgboost import XGBClassifier

logger = LoggerFactory().getLogger(__name__)
warnings.filterwarnings("ignore")


class ModelResult(NamedTuple):
    oof_preds: np.ndarray
    preds: Optional[np.ndarray]
    models: Dict[str, any]
    scores: Dict[str, float]


class XGBTrainer:
    def __init__(self, n_fold: int, metric: Callable):
        self.metric = metric
        self.n_fold = n_fold
        self.result = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
        verbose: Union[int, bool] = False,
    ) -> bool:
        models = dict()
        scores = dict()

        kf = StratifiedKFold(n_splits=self.n_fold, random_state=42, shuffle=True)
        splits = kf.split(X, y)
        xgb_oof = np.zeros((X.shape[0], 61))
        run = neptune.init(
            project="ds-wook/ai-hackaton", tags=["XGBoost", "Stratified KFold"]
        )
        for fold, (train_idx, valid_idx) in enumerate(splits):
            print("Fold :", fold)
            neptune_callback = NeptuneCallback(
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
            xgb_oof[valid_idx, :] = model.predict_proba(X_valid)

            models[f"fold_{fold}"] = model
            score = self.metric(y_valid, xgb_oof[valid_idx, :])
            scores[f"fold_{fold}"] = score
            logger.info(f"fold {fold}: {score}")

            gc.collect()

        oof_score = self.metric(y.values, xgb_oof)
        logger.info(f"oof score: {oof_score}")

        self.result = ModelResult(
            oof_preds=xgb_oof,
            models=models,
            preds=None,
            scores={
                "oof_score": oof_score,
                "KFoldsScores": scores,
            },
        )
        return True

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        folds = self.n_fold
        xgb_preds = np.zeros((X_test.shape[0], 61))

        for fold in range(folds):
            model = self.result.models[f"fold_{fold}"]
            xgb_preds += model.predict_proba(X_test) / self.n_fold

        return xgb_preds
