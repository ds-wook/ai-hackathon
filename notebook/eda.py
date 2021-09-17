# %%
from typing import Tuple

import numpy as np
import pandas as pd


def load_dataset(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train = pd.read_csv(path + "train_features.csv")
    y_train = pd.read_csv(path + "train_labels.csv")
    X_test = pd.read_csv(path + "test_features.csv")
    X_pivot_train = pd.pivot_table(
        data=X_train,
        values=X_train.columns[2:],
        index="id",
        aggfunc=["sum", "mean", "median", "min", "max", "std", "var"],
    )

    X_pivot_test = pd.pivot_table(
        data=X_test,
        values=X_test.columns[2:],
        index="id",
        aggfunc=["sum", "mean", "median", "min", "max", "std", "var"],
    )
    X_columns = [agg + "_" + column for agg, column in X_pivot_train.columns]
    X_pivot_train.columns = X_columns
    X_pivot_test.columns = X_columns
    X_pivot_train = X_pivot_train.reset_index()
    X_pivot_test = X_pivot_test.reset_index()
    train_data = pd.merge(X_pivot_train, y_train.loc[:, ["id", "label"]], on="id")
    train_data.label = train_data.label.astype("category")

    return train_data, X_pivot_test


path = "../input/ai-hackaton/"
train, test = load_dataset(path)

# %%
train.head()
# %%
train["label"]
# %%
train.label.value_counts()
# %%
