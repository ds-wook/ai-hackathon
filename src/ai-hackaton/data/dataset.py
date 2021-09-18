from itertools import combinations
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
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

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
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


def count_stop_moving(data: pd.DataFrame):
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
    func = [
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
    ]
    agg_dict = {col: func for col in train_x.columns[2:]}
    agg_train_x = train_x.groupby(["id"]).agg(agg_dict).reset_index()
    agg_test_x = test_x.groupby(["id"]).agg(agg_dict).reset_index()
    columns = ["id"] + [agg + "_" + name for name, agg in agg_train_x.columns][1:]
    agg_train_x.columns = columns
    agg_test_x.columns = columns

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

    func = [
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
    ]
    agg_dict = {col: func for col in train_x.columns[2:]}
    smoothed_agg_train_x = smoothed_train_x.groupby(["id"]).agg(agg_dict).reset_index()
    smoothed_agg_test_x = smoothed_test_x.groupby(["id"]).agg(agg_dict).reset_index()

    columns = ["id"] + [
        "smoothed" + "_" + agg + "_" + name
        for agg, name in smoothed_agg_train_x.columns
    ][1:]
    smoothed_agg_train_x.columns = columns
    smoothed_agg_test_x.columns = columns

    count_stop_moving_train = count_stop_moving(train_x)
    count_stop_moving_test = count_stop_moving(test_x)

    fft_train, fft_test = ft_trans("acc_xyz", train_x, test_x)

    total_train_x = pd.concat(
        [
            agg_train_x,
            smoothed_agg_train_x,
            count_stop_moving_train,
            fft_train,
        ],
        axis=1,
    ).drop(columns=["id"])

    total_test_x = pd.concat(
        [
            agg_test_x,
            smoothed_agg_test_x,
            count_stop_moving_test,
            fft_test,
        ],
        axis=1,
    ).drop(columns=["id"])

    return total_train_x, total_test_x
