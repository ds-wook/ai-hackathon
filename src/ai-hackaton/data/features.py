from typing import Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from shap import TreeExplainer
from tqdm import tqdm


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


def count_stop_moving(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["acc_x", "acc_y", "acc_z", "gy_x", "gy_y", "gy_z"]
    new_columns = ["count_stop_moving_" + x for x in columns]
    now_cum_data = pd.DataFrame()

    for user_id in tqdm(data.id.unique()):
        user_values = [user_id]

        for column in columns:
            target_data = data.query("id == @user_id")[[column]]
            cumsum_data = np.cumsum(target_data).iloc[:-1]
            target_data = target_data.iloc[1:]

            target_data.index = range(len(target_data))
            cumsum_data.index = range(len(cumsum_data))
            cumsum_data.columns = ["cum_" + column]

            concated = pd.concat([target_data, cumsum_data], axis=1)

            concated["if_changed"] = concated.apply(
                lambda x: 1 if (abs(x["cum_" + column]) - abs(x[column])) <= 0 else 0,
                axis=1,
            )

            if concated.query("if_changed == 1").shape[0] > 0:
                target_value = (
                    concated.query("if_changed == 1")
                    .apply(
                        lambda x: 0 if x["cum_" + column] * x[column] > 0 else 1, axis=1
                    )
                    .sum()
                )
                user_values.append(target_value)
            else:
                user_values.append(0)
        now_cum_data = now_cum_data.append([user_values])
    now_cum_data.columns = ["id"] + new_columns

    return now_cum_data
