from itertools import combinations
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)
from tqdm import tqdm


class KMeansFeaturizer:
    def __init__(
        self, k: int = 100, target_scale: int = 5, random_state: Optional[int] = None
    ):
        self.k = k
        self.target_scale = target_scale
        self.random_state = random_state
        self.cluster_encoder = OneHotEncoder().fit(np.array(range(k)).reshape(-1, 1))

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> object:
        if y is None:
            # No target variable, just do plain k-means
            km_model = KMeans(
                n_clusters=self.k, n_init=20, random_state=self.random_state
            )
            km_model.fit(X)

            self.km_model_ = km_model
            self.cluster_centers_ = km_model.cluster_centers_
            return self

        data_with_target = np.hstack((X, y[:, np.newaxis] * self.target_scale))

        km_model_pretrain = KMeans(
            n_clusters=self.k, n_init=20, random_state=self.random_state
        )
        km_model_pretrain.fit(data_with_target)
        km_model = KMeans(
            n_clusters=self.k,
            init=km_model_pretrain.cluster_centers_[:, : data_with_target.shape[1] - 1],
            n_init=1,
            max_iter=1,
        )
        km_model.fit(X)

        self.km_model = km_model
        self.cluster_centers_ = km_model.cluster_centers_
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        clusters = self.km_model.predict(X)
        onehot = self.cluster_encoder.transform(clusters.reshape(-1, 1)).toarray()
        max_col = onehot.shape[1]
        pca = PCA(n_components=max_col, random_state=0).fit(onehot)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        num_col = np.argmax(cumsum >= 0.99) + 1
        if num_col == 1:
            num_col = max_col
        pca = PCA(n_components=num_col, random_state=0).fit_transform(onehot)
        return pd.DataFrame(pca)

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X, y)


def ft_trans(name: str, train: pd.DataFrame, test: pd.DataFrame):
    def train_test(check, num_col):

        if check == "train":
            df_checking = train.copy()
            train_datas = np.zeros((len(df_checking.id.unique()), 304))

        elif check == "test":
            df_checking = test.copy()
            train_datas = np.zeros((len(df_checking.id.unique()), 304))

        for i, num in enumerate(tqdm(df_checking.id.unique())):
            tt = (
                df_checking.loc[df_checking.id == num][name]
                - df_checking.loc[df_checking.id == num][name].mean()
            )
            fmax = 50  # sampling frequency 1000 Hz
            dt = 1 / fmax  # sampling period
            N = 600  # length of signal

            # t = np.arange(0, N) * dt  # time = [0, dt, ..., (N-1)*dt]
            x = tt.values
            df = fmax / N  # df = 1/N = fmax/N
            f = np.arange(0, N) * df  # frq = [0, df, ..., (N-1)*df]
            xf = np.fft.fft(x) * dt
            tq_index = f[0 : int(N / 2 + 1)]
            tq_abs = np.abs(xf[0 : int(N / 2 + 1)])

            results = (
                pd.DataFrame(tq_abs, tq_index)
                .reset_index()
                .rename(columns={"index": "hz", 0: "abs_value"})
            )

            ar0 = np.array([num])
            ar1 = results.abs_value.values
            ar2 = np.array(
                [skew(results.abs_value), kurtosis(results.abs_value, fisher=True)]
            )
            return_value = np.concatenate([ar0, ar1, ar2])
            train_datas[i] = return_value

        return train_datas

    col_ft = ["_" + str(x) for x in range(304)]

    num_col = len(col_ft)
    train_datas = train_test("train", num_col)
    test_datas = train_test("test", num_col)

    col_ft_F = ["id"] + [name + "_" + x for x in col_ft[1:]]
    train_df = pd.DataFrame(train_datas, columns=col_ft_F)
    test_df = pd.DataFrame(test_datas, columns=col_ft_F)

    train_df.id = train_df.id.astype("int")
    test_df.id = test_df.id.astype("int")

    return train_df, test_df


def time_log_data(data: pd.DataFrame, data_name: str, num: int, user_id: int, t: int):
    target_time = float(num[0] + "." + num[1:])
    try:
        globals()["log_" + num + "_" + data_name][user_id][t] = (
            data[user_id][t + int(target_time // 0.02)] - data[user_id][t]
        )
    except:
        pass


def make_dataset(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_x = pd.read_csv(path + "train_features.csv")
    train_y = pd.read_csv(path + "train_labels.csv")
    test_x = pd.read_csv(path + "test_features.csv")
    train = pd.merge(train_x, train_y, on="id")
    train = (
        train.groupby(["label", "time"]).agg("mean").reset_index().drop(columns=["id"])
    )
    label_name = (
        train_y.iloc[:, 1:]
        .sort_values(by="label")
        .drop_duplicates()
        .drop(columns=["label"])
    )
    label_name.index = range(61)
    label_name = label_name.to_dict()["label_desc"]

    acc_columns = train_x.columns[2:5]
    gy_columns = train_x.columns[5:]

    for target_list in [acc_columns, gy_columns]:
        for a, b in combinations(target_list, 2):
            column_name = a + b[-1]
            train_x[column_name] = (train_x[a] ** 2 + train_x[b] ** 2) ** (1 / 2)
    train_x["acc_xyz"] = (
        train_x["acc_x"] ** 2 + train_x["acc_y"] ** 2 + train_x["acc_z"] ** 2
    ) ** (1 / 3)
    train_x["gy_xyz"] = (
        train_x["gy_x"] ** 2 + train_x["gy_y"] ** 2 + train_x["gy_z"] ** 2
    ) ** (1 / 3)

    for target_list in [acc_columns, gy_columns]:
        for a, b in combinations(target_list, 2):
            column_name = a + b[-1]
            test_x[column_name] = (test_x[a] ** 2 + test_x[b] ** 2) ** (1 / 2)
    test_x["acc_xyz"] = (
        test_x["acc_x"] ** 2 + test_x["acc_y"] ** 2 + test_x["acc_z"] ** 2
    ) ** (1 / 3)
    test_x["gy_xyz"] = (
        test_x["gy_x"] ** 2 + test_x["gy_y"] ** 2 + test_x["gy_z"] ** 2
    ) ** (1 / 3)
    agg_train_x = pd.pivot_table(
        train_x,
        values=train_x.columns[2:],
        index="id",
        aggfunc=[
            "sum",
            "mean",
            "mad",
            "median",
            "min",
            "max",
            "std",
            "var",
            "sem",
            "skew",
            "quantile",
        ],
    ).reset_index()
    print("==TRAIN DATA DONE ==")
    agg_test_x = pd.pivot_table(
        test_x,
        values=test_x.columns[2:],
        index="id",
        aggfunc=[
            "sum",
            "mean",
            "mad",
            "median",
            "min",
            "max",
            "std",
            "var",
            "sem",
            "skew",
            "quantile",
        ],
    ).reset_index()

    columns = ["id"] + [agg + "_" + name for agg, name in agg_train_x.columns][1:]
    agg_train_x.columns = columns
    agg_test_x.columns = columns
    agg_train_x.to_csv(path + "features/agg_train.csv", index=False)
    agg_test_x.to_csv(path + "features/agg_test.csv", index=False)

    grouped_train_x = train_x.iloc[:, 2:].values.reshape(-1, 600, train_x.shape[1] - 2)
    grouped_test_x = test_x.iloc[:, 2:].values.reshape(-1, 600, test_x.shape[1] - 2)

    for num in ["002", "010", "020", "030", "050", "100", "200", "300", "600"]:
        target_time = float(num[0] + "." + num[1:])
        globals()["log_" + num + "_train_x"] = np.zeros(
            (
                grouped_train_x.shape[0],
                grouped_train_x.shape[1] - int(target_time // 0.02),
                grouped_train_x.shape[2],
            )
        )
        globals()["log_" + num + "_test_x"] = np.zeros(
            (
                grouped_test_x.shape[0],
                grouped_test_x.shape[1] - int(target_time // 0.02),
                grouped_test_x.shape[2],
            )
        )

    for data, data_name in zip(
        [grouped_train_x, grouped_test_x], ["train_x", "test_x"]
    ):
        for user_id in tqdm(range(data.shape[0])):
            for t in range(599):
                for num in [
                    "002",
                    "010",
                    "020",
                    "030",
                    "050",
                    "100",
                    "200",
                    "300",
                    "600",
                ]:
                    time_log_data(data, data_name, num, user_id, t)
    print("== Making Log Data Done ==")
    columns = ["id"] + list(train_x.columns[2:])

    for data_name in ["train_x", "test_x"]:
        for num in ["002", "010", "020", "030", "050", "100", "200", "300", "600"]:
            log_num = globals()["log_" + num + "_" + data_name].shape[1]
            globals()["log_" + num + "_" + data_name] = pd.DataFrame(
                globals()["log_" + num + "_" + data_name].reshape(-1, 14)
            ).reset_index()
            globals()["log_" + num + "_" + data_name].columns = columns
            globals()["log_" + num + "_" + data_name].id = globals()[
                "log_" + num + "_" + data_name
            ].id.apply(lambda x: x // log_num)
    print("==Log Data to DF done==")

    for num in tqdm(["002", "010", "020", "030", "050", "100", "200", "300", "600"]):
        globals()["log_" + num + "_train_x"] = pd.pivot_table(
            globals()["log_" + num + "_train_x"],
            values=globals()["log_" + num + "_train_x"].columns[1:],
            index="id",
            aggfunc=[
                "sum",
                "mean",
                "mad",  # 합, 평균, 평균 절대 편차
                "median",
                "min",
                "max",  # 중앙값, 최소값, 최대값
                "std",
                "var",  # 베셀 보정 표본 표준편차, 비편향 편차
                "sem",
                "skew",
                "quantile",
            ],  # 평균의 표준오차, 표본왜도, 포본분위수
        ).reset_index()

        globals()["log_" + num + "_test_x"] = pd.pivot_table(
            globals()["log_" + num + "_test_x"],
            values=globals()["log_" + num + "_test_x"].columns[1:],
            index="id",
            aggfunc=[
                "sum",
                "mean",
                "mad",
                "median",
                "min",
                "max",
                "std",
                "var",
                "sem",
                "skew",
                "quantile",
            ],
        ).reset_index()
        columns = ["id"] + [
            "log_change_" + num + "_" + agg + "_" + name
            for agg, name in globals()["log_" + num + "_train_x"].columns
        ][1:]
        globals()["log_" + num + "_train_x"].columns = columns
        globals()["log_" + num + "_test_x"].columns = columns

    log_change_train = pd.DataFrame(list(range(3125)), columns=["id"])
    log_change_test = pd.DataFrame(list(range(3125, 3907)), columns=["id"])

    for num in ["002", "010", "020", "030", "050", "100", "200", "300", "600"]:
        log_change_train = pd.merge(
            log_change_train, globals()["log_" + num + "_train_x"]
        )
        globals()["log_" + num + "_test_x"].id = (
            globals()["log_" + num + "_test_x"].id + 3125
        )
        log_change_test = pd.merge(log_change_test, globals()["log_" + num + "_test_x"])
    print("==Log CHANGE DF to Pivot Table Done==")
    log_change_train.to_csv(path + "features/log_change_train.csv", index=False)
    log_change_test.to_csv(path + "features/log_change_test.csv", index=False)

    columns = ["id"] + list(train_x.columns[2:])
    for num in tqdm(["002", "010", "020", "030", "050", "100", "200", "300", "600"]):
        target_time = float(num[0] + "." + num[1:])
        skip_num = int(num) // 2
        target_index = list(range(0, 600, skip_num)) + [599]
        globals()["log_" + num + "_train_x"] = pd.DataFrame(
            grouped_train_x[:, target_index, :].reshape(-1, 14)
        ).reset_index()
        globals()["log_" + num + "_train_x"].columns = columns
        globals()["log_" + num + "_train_x"].id = globals()[
            "log_" + num + "_train_x"
        ].id.apply(lambda x: x // len(target_index))
        globals()["log_" + num + "_test_x"] = pd.DataFrame(
            grouped_test_x[:, target_index, :].reshape(-1, 14)
        ).reset_index()
        globals()["log_" + num + "_test_x"].columns = columns
        globals()["log_" + num + "_test_x"].id = globals()[
            "log_" + num + "_test_x"
        ].id.apply(lambda x: x // len(target_index))

    for num in tqdm(["002", "010", "020", "030", "050", "100", "200", "300", "600"]):
        globals()["log_" + num + "_train_x"] = pd.pivot_table(
            globals()["log_" + num + "_train_x"],
            values=globals()["log_" + num + "_train_x"].columns[1:],
            index="id",
            aggfunc=[
                "sum",
                "mean",
                "mad",  # 합, 평균, 평균 절대 편차
                "median",
                "min",
                "max",  # 중앙값, 최소값, 최대값
                "std",
                "var",  # 베셀 보정 표본 표준편차, 비편향 편차
                "sem",
                "skew",
                "quantile",
            ],  # 평균의 표준오차, 표본왜도, 포본분위수
        ).reset_index()

        globals()["log_" + num + "_test_x"] = pd.pivot_table(
            globals()["log_" + num + "_test_x"],
            values=globals()["log_" + num + "_test_x"].columns[1:],
            index="id",
            aggfunc=[
                "sum",
                "mean",
                "mad",
                "median",
                "min",
                "max",
                "std",
                "var",
                "sem",
                "skew",
                "quantile",
            ],
        ).reset_index()
        columns = ["id"] + [
            "log_skip_" + num + "_" + agg + "_" + name
            for agg, name in globals()["log_" + num + "_train_x"].columns
        ][1:]
        globals()["log_" + num + "_train_x"].columns = columns
        globals()["log_" + num + "_test_x"].columns = columns

    log_skip_train = pd.DataFrame(list(range(3125)), columns=["id"])
    log_skip_test = pd.DataFrame(list(range(3125, 3907)), columns=["id"])
    for num in ["002", "010", "020", "030", "050", "100", "200", "300", "600"]:
        log_skip_train = pd.merge(log_skip_train, globals()["log_" + num + "_train_x"])
        globals()["log_" + num + "_test_x"].id = (
            globals()["log_" + num + "_test_x"].id + 3125
        )
        log_skip_test = pd.merge(log_skip_test, globals()["log_" + num + "_test_x"])

    log_skip_train.to_csv("features/log_skip_train.csv", index=False)
    log_skip_test.to_csv("features/log_skip_test.csv", index=False)

    smoothed_train_x = pd.DataFrame()
    smoothed_test_x = pd.DataFrame()

    for user_id in tqdm(range(train_x["id"].nunique())):
        temp = train_x.query("id == @user_id")
        temp.iloc[:, 2:] = gaussian_filter1d(temp.iloc[:, 2:], axis=0, sigma=10)
        smoothed_train_x = pd.concat([smoothed_train_x, temp])

    for user_id in tqdm(range(test_x["id"].nunique())):
        temp = test_x.query("id == @user_id")
        temp.iloc[:, 2:] = gaussian_filter1d(temp.iloc[:, 2:], axis=0, sigma=10)
        smoothed_test_x = pd.concat([smoothed_test_x, temp])

    smoothed_agg_train_x = pd.pivot_table(
        train_x,
        values=train_x.columns[2:],
        index="id",
        aggfunc=[
            "sum",
            "mean",
            "mad",  # 합, 평균, 평균 절대 편차
            "median",
            "min",
            "max",  # 중앙값, 최소값, 최대값
            "std",
            "var",  # 베셀 보정 표본 표준편차, 비편향 편차
            "sem",
            "skew",
            "quantile",
        ],  # 평균의 표준오차, 표본왜도, 포본분위수
    ).reset_index()
    print("==TRAIN DATA DONE ==")
    smoothed_agg_test_x = pd.pivot_table(
        test_x,
        values=test_x.columns[2:],
        index="id",
        aggfunc=[
            "sum",
            "mean",
            "mad",
            "median",
            "min",
            "max",
            "std",
            "var",
            "sem",
            "skew",
            "quantile",
        ],
    ).reset_index()
    print("==TEST DATA DONE ==")
    columns = ["id"] + [
        "smoothed" + "_" + agg + "_" + name
        for agg, name in smoothed_agg_train_x.columns
    ][1:]
    smoothed_agg_train_x.columns = columns
    smoothed_agg_test_x.columns = columns

    smoothed_agg_train_x.to_csv(path + "features/smoothed_agg_train.csv", index=False)
    smoothed_agg_test_x.to_csv(path + "features/smoothed_agg_test.csv", index=False)

    columns = ["acc_x", "acc_y", "acc_z", "gy_x", "gy_y", "gy_z"]
    new_columns = ["count_stop_moving_" + x for x in columns]
    now_cum_data = pd.DataFrame()

    for user_id in tqdm(train_x.id.unique()):
        user_values = [user_id]
        for column in columns:

            target_data = train_x.query("id == @user_id")[[column]]
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

    now_cum_data.to_csv(path + "features/count_stop_moving_train.csv", index=False)

    columns = ["acc_x", "acc_y", "acc_z", "gy_x", "gy_y", "gy_z"]
    new_columns = ["count_stop_moving_" + x for x in columns]
    now_cum_data = pd.DataFrame()

    for user_id in tqdm(test_x.id.unique()):
        user_values = [user_id]
        for column in columns:

            target_data = test_x.query("id == @user_id")[[column]]
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
    now_cum_data.to_csv(path + "features/count_stop_moving_test.csv", index=False)

    train_fft, test_fft = ft_trans("acc_xyz", train_x, test_x)
    train_fft.to_csv(path + "features/fft_train.csv", index=False)
    test_fft.to_csv(path + "features/fft_test.csv", index=False)
