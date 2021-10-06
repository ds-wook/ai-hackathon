from typing import Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from shap import TreeExplainer


def select_features(
    train: pd.DataFrame, label: pd.Series, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    model = LGBMClassifier(random_state=42)
    print(f"{model.__class__.__name__} Train Start!")
    model.fit(train, label)
    explainer = TreeExplainer(model)

    shap_values = explainer.shap_values(test)
    shap_sum = np.abs(shap_values).mean(axis=1).sum(axis=0)

    importance_df = pd.DataFrame([test.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ["column_name", "shap_importance"]

    importance_df = importance_df.sort_values("shap_importance", ascending=False)
    print(
        f'{str(np.round(importance_df.query("shap_importance == 0").shape[0] / importance_df.shape[0], 3)* 100)[:4]}%의 Importance가 0인 Feature가 존재'
    )

    importance_df = importance_df.query("shap_importance != 0")
    boosting_shap_col = importance_df.column_name.values.tolist()
    print(f"총 {len(boosting_shap_col)}개 선택")

    shap_train = train.loc[:, boosting_shap_col]
    shap_test = test.loc[:, boosting_shap_col]

    return shap_train, shap_test
