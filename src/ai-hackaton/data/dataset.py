import math
import warnings
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans

pi = math.pi
pd.options.display.max_columns = 500
warnings.filterwarnings("ignore")


def range_func(x: List[Union[int, float]]) -> float:
    max_val = np.max(x)
    min_val = np.min(x)
    range_val = max_val - min_val
    return range_val


def iqr_func2(x: List[Union[int, float]]) -> float:
    q3, q1 = np.percentile(x, [20, 80])
    iqr = q3 - q1
    return iqr


def iqr_func3(x: List[Union[int, float]]) -> float:
    q3, q1 = np.percentile(x, [40, 60])
    iqr = q3 - q1
    return iqr


def iqr_func4(x: List[Union[int, float]]) -> float:
    q3, q1 = np.percentile(x, [15, 95])
    iqr = q3 - q1
    return iqr


def premad(x: List[Union[int, float]]) -> float:
    return np.median(np.absolute(x - np.median(x, axis=0)), axis=0)


def preskew(x: List[Union[int, float]]) -> float:
    return skew(x)


def prekurt(x: List[Union[int, float]]) -> float:
    return kurtosis(x, fisher=True)


def load_dataset(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = "../input/ai-hackaton/"
    train = pd.read_csv(path + "train_features.csv")
    test = pd.read_csv(path + "test_features.csv")
    change_train = train.drop("time", axis=1)
    train_change = change_train.groupby("id").diff().reset_index().fillna(0)
    change_test = test.drop("time", axis=1)
    test_change = change_test.groupby("id").diff().reset_index().fillna(0)
    train_change.rename(columns={"index": "id"}, inplace=True)
    test_change.rename(columns={"index": "id"}, inplace=True)
    train["acc_vector"] = np.sqrt(
        (train["acc_x"] ** 2) + (train["acc_y"] ** 2) + (train["acc_z"] ** 2)
    )
    train["gy_vector"] = np.sqrt(
        (train["gy_x"] ** 2) + (train["gy_y"] ** 2) + (train["gy_z"] ** 2)
    )

    test["acc_vector"] = np.sqrt(
        (test["acc_x"] ** 2) + (test["acc_y"] ** 2) + (test["acc_z"] ** 2)
    )
    test["gy_vector"] = np.sqrt(
        (test["gy_x"] ** 2) + (test["gy_y"] ** 2) + (test["gy_z"] ** 2)
    )

    train["acc_YZvector"] = np.sqrt((train["acc_y"] ** 2) + (train["acc_z"] ** 2))
    train["gy_YZvector"] = np.sqrt((train["gy_y"] ** 2) + (train["gy_z"] ** 2))

    train["acc_XYvector"] = np.sqrt((train["acc_x"] ** 2) + (train["acc_y"] ** 2))
    train["gy_XYvector"] = np.sqrt((train["gy_x"] ** 2) + (train["gy_y"] ** 2))

    train["acc_XZvector"] = np.sqrt((train["acc_x"] ** 2) + (train["acc_z"] ** 2))
    train["gy_XZvector"] = np.sqrt((train["gy_x"] ** 2) + (train["gy_z"] ** 2))

    test["acc_YZvector"] = np.sqrt((test["acc_y"] ** 2) + (test["acc_z"] ** 2))
    test["gy_YZvector"] = np.sqrt((test["gy_y"] ** 2) + (test["gy_z"] ** 2))

    test["acc_XYvector"] = np.sqrt((test["acc_x"] ** 2) + (test["acc_y"] ** 2))
    test["gy_XYvector"] = np.sqrt((test["gy_x"] ** 2) + (test["gy_y"] ** 2))

    test["acc_XZvector"] = np.sqrt((test["acc_x"] ** 2) + (test["acc_z"] ** 2))
    test["gy_XZvector"] = np.sqrt((test["gy_x"] ** 2) + (test["gy_z"] ** 2))

    # 자이로스코프 무게중심
    train["gy_Centerofgravity"] = (train["gy_x"] + train["gy_y"] + train["gy_z"]) / 3
    test["gy_Centerofgravity"] = (test["gy_x"] + test["gy_y"] + test["gy_z"]) / 3
    # roll & pitch
    train["roll"] = np.arctan(
        train["acc_y"] / np.sqrt(train["acc_x"] ** 2 + train["acc_z"] ** 2)
    )
    test["roll"] = np.arctan(
        test["acc_y"] / np.sqrt(test["acc_x"] ** 2 + test["acc_z"] ** 2)
    )

    train["pitch"] = np.arctan(
        train["acc_x"] / np.sqrt(train["acc_y"] ** 2 + train["acc_z"] ** 2)
    )
    test["pitch"] = np.arctan(
        test["acc_x"] / np.sqrt(test["acc_y"] ** 2 + test["acc_z"] ** 2)
    )

    train["math_roll"] = np.arctan(
        -train["acc_x"] / np.sqrt(train["acc_y"] ** 2 + train["acc_z"] ** 2)
    ) * (180 / pi)
    train["math_pitch"] = np.arctan(
        train["acc_y"] / np.sqrt(train["acc_x"] ** 2 + train["acc_z"] ** 2)
    ) * (180 / pi)

    test["math_roll"] = np.arctan(
        -test["acc_x"] / np.sqrt(test["acc_y"] ** 2 + test["acc_z"] ** 2)
    ) * (180 / pi)
    test["math_pitch"] = np.arctan(
        test["acc_y"] / np.sqrt(test["acc_x"] ** 2 + test["acc_z"] ** 2)
    ) * (180 / pi)

    train["gy_roll"] = np.arctan(
        train["gy_y"] / np.sqrt(train["gy_x"] ** 2 + train["gy_z"] ** 2)
    )
    test["gy_roll"] = np.arctan(
        test["gy_y"] / np.sqrt(test["gy_x"] ** 2 + test["gy_z"] ** 2)
    )

    train["gy_pitch"] = np.arctan(
        train["gy_x"] / np.sqrt(train["gy_y"] ** 2 + train["gy_z"] ** 2)
    )
    test["gy_pitch"] = np.arctan(
        test["gy_x"] / np.sqrt(test["gy_y"] ** 2 + test["gy_z"] ** 2)
    )

    train["gy_math_roll"] = np.arctan(
        -train["gy_x"] / np.sqrt(train["gy_y"] ** 2 + train["gy_z"] ** 2)
    ) * (180 / pi)
    train["gy_math_pitch"] = np.arctan(
        train["gy_y"] / np.sqrt(train["gy_x"] ** 2 + train["gy_z"] ** 2)
    ) * (180 / pi)

    test["gy_math_roll"] = np.arctan(
        -test["gy_x"] / np.sqrt(test["gy_y"] ** 2 + test["gy_z"] ** 2)
    ) * (180 / pi)
    test["gy_math_pitch"] = np.arctan(
        test["gy_y"] / np.sqrt(test["gy_x"] ** 2 + test["gy_z"] ** 2)
    ) * (180 / pi)

    features = [
        "id",
        "acc_x",
        "acc_y",
        "acc_z",
        "gy_x",
        "gy_y",
        "gy_z",
        "acc_vector",
        "gy_vector",
        "acc_YZvector",
        "gy_YZvector",
        "acc_XYvector",
        "gy_XYvector",
        "acc_XZvector",
        "gy_XZvector",
        "gy_Centerofgravity",
    ]
    features2 = [
        "id",
        "roll",
        "pitch",
        "math_roll",
        "math_pitch",
        "gy_roll",
        "gy_pitch",
        "gy_math_roll",
        "gy_math_pitch",
    ]

    train_preprocess = (
        train[features]
        .groupby("id")
        .agg(
            [
                "max",
                "min",
                "mean",
                "std",
                "median",
                range_func,
                iqr_func2,
                iqr_func3,
                iqr_func4,
                premad,
                preskew,
                prekurt,
            ]
        )
    )
    temp_train_preprocess = (
        train[features2]
        .groupby("id")
        .agg([range_func, iqr_func2, iqr_func3, iqr_func4, premad, preskew, prekurt])
    )
    test_preprocess = (
        test[features]
        .groupby("id")
        .agg(
            [
                "max",
                "min",
                "mean",
                "std",
                "median",
                range_func,
                iqr_func2,
                iqr_func3,
                iqr_func4,
                premad,
                preskew,
                prekurt,
            ]
        )
    )
    temp_test_preprocess = (
        test[features2]
        .groupby("id")
        .agg([range_func, iqr_func2, iqr_func3, iqr_func4, premad, preskew, prekurt])
    )

    train_preprocess = pd.concat([train_preprocess, temp_train_preprocess], axis=1)
    test_preprocess = pd.concat([test_preprocess, temp_test_preprocess], axis=1)

    train_preprocess.columns = [i[0] + "_" + i[1] for i in train_preprocess.columns]
    test_preprocess.columns = [i[0] + "_" + i[1] for i in test_preprocess.columns]

    train_preprocess["acc_std_mean"] = (
        train_preprocess["acc_x_std"]
        + train_preprocess["acc_y_std"]
        + train_preprocess["acc_z_std"]
    ) / 3
    train_preprocess["gy_std_mean"] = (
        train_preprocess["gy_x_std"]
        + train_preprocess["gy_y_std"]
        + train_preprocess["gy_z_std"]
    ) / 3

    test_preprocess["acc_std_mean"] = (
        test_preprocess["acc_x_std"]
        + test_preprocess["acc_y_std"]
        + test_preprocess["acc_z_std"]
    ) / 3
    test_preprocess["gy_std_mean"] = (
        test_preprocess["gy_x_std"]
        + test_preprocess["gy_y_std"]
        + test_preprocess["gy_z_std"]
    ) / 3

    # 가속도계 첫번째 데이터
    train_acc_head1 = (
        train.groupby(["id"])[["acc_x", "acc_y", "acc_z"]].first().reset_index()
    )
    train_acc_head1.columns = ["id", "first_acc_x", "first_acc_y", "first_acc_z"]
    train_acc_head1.set_index("id", inplace=True)

    test_acc_head1 = (
        test.groupby(["id"])[["acc_x", "acc_y", "acc_z"]].first().reset_index()
    )
    test_acc_head1.columns = ["id", "first_acc_x", "first_acc_y", "first_acc_z"]
    test_acc_head1.set_index("id", inplace=True)

    # 가속도계 첫 3초
    train_acc_head = (
        train.loc[:, ["id", "time", "acc_x", "acc_y", "acc_z"]][train["time"] < 150]
        .drop("time", axis=1)
        .groupby("id")
        .mean()
        .reset_index()
    )
    train_acc_head.columns = ["id", "head_acc_x", "head_acc_y", "head_acc_z"]
    train_acc_head = train_acc_head.groupby("id").mean()

    test_acc_head = (
        test.loc[:, ["id", "time", "acc_x", "acc_y", "acc_z"]][train["time"] < 150]
        .drop("time", axis=1)
        .groupby("id")
        .mean()
        .reset_index()
    )
    test_acc_head.columns = ["id", "head_acc_x", "head_acc_y", "head_acc_z"]
    test_acc_head = test_acc_head.groupby("id").mean()

    train_preprocess = pd.concat(
        [train_preprocess, train_acc_head, train_acc_head1], axis=1
    )
    test_preprocess = pd.concat(
        [test_preprocess, test_acc_head, test_acc_head1], axis=1
    )

    # 자이로스코프 첫 3초
    train_gy_head = (
        train.loc[:, ["id", "time", "gy_x", "gy_y", "gy_z"]][train["time"] < 150]
        .drop("time", axis=1)
        .groupby("id")
        .mean()
        .reset_index()
    )
    train_gy_head.columns = ["id", "head_gy_x", "head_gy_y", "head_gy_z"]
    train_gy_head = train_gy_head.groupby("id").mean()

    test_gy_head = (
        test.loc[:, ["id", "time", "gy_x", "gy_y", "gy_z"]][train["time"] < 150]
        .drop("time", axis=1)
        .groupby("id")
        .mean()
        .reset_index()
    )
    test_gy_head.columns = ["id", "head_gy_x", "head_gy_y", "head_gy_z"]
    test_gy_head = test_gy_head.groupby("id").mean()

    train_preprocess = pd.concat([train_preprocess, train_gy_head], axis=1)
    test_preprocess = pd.concat([test_preprocess, test_gy_head], axis=1)

    model = KMeans(n_clusters=5, random_state=20)
    model.fit(train_preprocess)
    train_predict = model.predict(train_preprocess)
    train_preprocess["cluster"] = train_predict

    test_predict = model.predict(test_preprocess)
    test_preprocess["cluster"] = test_predict

    column_name = list(train_preprocess.iloc[:, :247].columns)
    column_name.extend(
        [i[0] + "-" + i[1] for i in train_preprocess.iloc[:, 247:-1].columns]
    )
    column_name.extend(list(train_preprocess.iloc[:, -1:].columns))

    train_preprocess.columns = column_name
    test_preprocess.columns = column_name

    return train_preprocess, test_preprocess
