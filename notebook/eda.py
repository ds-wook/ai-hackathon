# %%
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy.fftpack import fft, fftfreq, ifft
from scipy.signal import medfilt
from tqdm import tqdm

path = "../input/ai-hackaton/"
train = pd.read_csv(path + "train_features.csv")
test = pd.read_csv(path + "test_features.csv")


def median(signal):  # input: numpy array 1D (one column)
    array = np.array(signal)
    # applying the median filter
    med_filtered = sp.signal.medfilt(
        array, kernel_size=3
    )  # applying the median filter order3(kernel_size=3)
    return med_filtered  # return the med-filtered signal: numpy array 1D


def jerk_one_signal(signal, dt):
    return np.array([(signal[i + 1] - signal[i]) / dt for i in range(len(signal) - 1)])


sampling_freq = 50
nyq = sampling_freq / float(
    2
)  # nyq is the nyquist frequency equal to the half of the sampling frequency[50/2= 25 Hz]

freq1 = (
    0.3  # freq1=0.3 hertz [Hz] the cuttoff frequency between the DC compoenents [0,0.3]
)
#           and the body components[0.3,20]hz
freq2 = 20  # freq2= 20 Hz the cuttoff frequcency between the body components[0.3,20] hz
#             and the high frequency noise components [20,25] hz

# Function name: components_selection_one_signal

# Inputs: t_signal:1D numpy array (time domain signal);

# Outputs: (total_component,t_DC_component , t_body_component, t_noise)
#           type(1D array,1D array, 1D array)

# cases to discuss: if the t_signal is an acceleration signal then the t_DC_component is the gravity component [Grav_acc]
#                   if the t_signal is a gyro signal then the t_DC_component is not useful
# t_noise component is not useful
# if the t_signal is an acceleration signal then the t_body_component is the body's acceleration component [Body_acc]
# if the t_signal is a gyro signal then the t_body_component is the body's angular velocity component [Body_gyro]


def components_selection_one_signal(t_signal, freq1, freq2):
    t_signal = np.array(t_signal)
    t_signal_length = len(t_signal)  # number of points in a t_signal

    # the t_signal in frequency domain after applying fft
    f_signal = fft(t_signal)  # 1D numpy array contains complex values (in C)

    # generate frequencies associated to f_signal complex values
    freqs = np.array(
        sp.fftpack.fftfreq(t_signal_length, d=1 / float(sampling_freq))
    )  # frequency values between [-25hz:+25hz]

    # DC_component: f_signal values having freq between [-0.3 hz to 0 hz] and from [0 hz to 0.3hz]
    #                                                             (-0.3 and 0.3 are included)

    # noise components: f_signal values having freq between [-25 hz to 20 hz[ and from ] 20 hz to 25 hz]
    #                                                               (-25 and 25 hz inculded 20hz and -20hz not included)

    # selecting body_component: f_signal values having freq between [-20 hz to -0.3 hz] and from [0.3 hz to 20 hz]
    #                                                               (-0.3 and 0.3 not included , -20hz and 20 hz included)

    f_DC_signal = []  # DC_component in freq domain
    f_body_signal = []  # body component in freq domain numpy.append(a, a[0])
    f_noise_signal = []  # noise in freq domain

    for i in range(len(freqs)):  # iterate over all available frequencies

        # selecting the frequency value
        freq = freqs[i]

        # selecting the f_signal value associated to freq
        value = f_signal[i]

        # Selecting DC_component values
        if abs(freq) > 0.3:  # testing if freq is outside DC_component frequency ranges
            f_DC_signal.append(
                float(0)
            )  # add 0 to  the  list if it was the case (the value should not be added)
        else:  # if freq is inside DC_component frequency ranges
            f_DC_signal.append(value)  # add f_signal value to f_DC_signal list

        # Selecting noise component values
        if abs(freq) <= 20:  # testing if freq is outside noise frequency ranges
            f_noise_signal.append(
                float(0)
            )  # # add 0 to  f_noise_signal list if it was the case
        else:  # if freq is inside noise frequency ranges
            f_noise_signal.append(value)  # add f_signal value to f_noise_signal

        # Selecting body_component values
        if (
            abs(freq) <= 0.3 or abs(freq) > 20
        ):  # testing if freq is outside Body_component frequency ranges
            f_body_signal.append(float(0))  # add 0 to  f_body_signal list
        else:  # if freq is inside Body_component frequency ranges
            f_body_signal.append(value)  # add f_signal value to f_body_signal list

    ################### Inverse the transformation of signals in freq domain ########################
    # applying the inverse fft(ifft) to signals in freq domain and put them in float format
    t_DC_component = ifft(np.array(f_DC_signal)).real
    t_body_component = ifft(np.array(f_body_signal)).real
    t_noise = ifft(np.array(f_noise_signal)).real

    total_component = (
        t_signal - t_noise
    )  # extracting the total component(filtered from noise)
    #  by substracting noise from t_signal (the original signal).

    # return outputs mentioned earlier
    return (total_component, t_DC_component, t_body_component, t_noise)


###################################################################################
# https://github.com/anas337/Human-Activity-Recognition-Using-Smartphones.github.io#
###################################################################################

for id in train.id.unique():
    for col in train.keys()[2:8]:
        median_value = median(np.array(train.loc[train.id == id, col]))
        train.loc[
            int(id) * 600 : (int(id) + 1) * 600 - 1, "median_" + str(col)
        ] = median_value

        if "acc" in col:
            _, grav, body, _ = components_selection_one_signal(median_value, 0.3, 20)
            body_jerk = jerk_one_signal(body, 12 / 600)
            train.loc[
                int(id) * 600 : (int(id) + 1) * 600 - 1, "grav_" + str(col)
            ] = grav
            train.loc[
                int(id) * 600 : (int(id) + 1) * 600 - 1, "body_" + str(col)
            ] = body
            train.loc[
                int(id) * 600 : (int(id) + 1) * 600 - 1, "body_jerk_" + str(col)
            ] = np.insert(body_jerk, 0, body_jerk[0])

        if "gy" in col:
            _, _, body, _ = components_selection_one_signal(median_value, 0.3, 20)
            body_jerk = jerk_one_signal(body, 12 / 600)
            train.loc[
                int(id) * 600 : (int(id) + 1) * 600 - 1, "body_" + str(col)
            ] = body
            train.loc[
                int(id) * 600 : (int(id) + 1) * 600 - 1, "body_jerk_" + str(col)
            ] = np.insert(body_jerk, 0, body_jerk[0])

train["acc_mag"] = (
    train["acc_x"] ** 2 + train["acc_y"] ** 2 + train["acc_z"] ** 2
) ** (0.5)
train["grav_acc_mag"] = (
    train["grav_acc_x"] ** 2 + train["grav_acc_y"] ** 2 + train["grav_acc_z"] ** 2
) ** (0.5)
train["body_acc_mag"] = (
    train["body_acc_x"] ** 2 + train["body_acc_y"] ** 2 + train["body_acc_z"] ** 2
) ** (0.5)
train["body_jerk_acc_mag"] = (
    train["body_jerk_acc_x"] ** 2
    + train["body_jerk_acc_y"] ** 2
    + train["body_jerk_acc_z"] ** 2
) ** (0.5)
train["gy_mag"] = (train["gy_x"] ** 2 + train["gy_y"] ** 2 + train["gy_z"] ** 2) ** (
    0.5
)
train["body_gy_mag"] = (
    train["body_gy_x"] ** 2 + train["body_gy_y"] ** 2 + train["body_gy_z"] ** 2
) ** (0.5)
train["body_jerk_gy_mag"] = (
    train["body_jerk_gy_x"] ** 2
    + train["body_jerk_gy_y"] ** 2
    + train["body_jerk_gy_z"] ** 2
) ** (0.5)

for idx, id in enumerate(test.id.unique()):

    for col in test.keys()[2:8]:
        median_value = median(np.array(test.loc[test.id == id, col]))
        test.loc[
            int(idx) * 600 : (int(idx) + 1) * 600 - 1, "median_" + str(col)
        ] = median_value

        if "acc" in col:
            _, grav, body, _ = components_selection_one_signal(median_value, 0.3, 20)
            body_jerk = jerk_one_signal(body, 12 / 600)
            test.loc[
                int(idx) * 600 : (int(idx) + 1) * 600 - 1, "grav_" + str(col)
            ] = grav
            test.loc[
                int(idx) * 600 : (int(idx) + 1) * 600 - 1, "body_" + str(col)
            ] = body
            test.loc[
                int(idx) * 600 : (int(idx) + 1) * 600 - 1, "body_jerk_" + str(col)
            ] = np.insert(body_jerk, 0, body_jerk[0])

        if "gy" in col:
            _, _, body, _ = components_selection_one_signal(median_value, 0.3, 20)
            body_jerk = jerk_one_signal(body, 12 / 600)
            test.loc[
                int(idx) * 600 : (int(idx) + 1) * 600 - 1, "body_" + str(col)
            ] = body
            test.loc[
                int(idx) * 600 : (int(idx) + 1) * 600 - 1, "body_jerk_" + str(col)
            ] = np.insert(body_jerk, 0, body_jerk[0])

test["acc_mag"] = (test["acc_x"] ** 2 + test["acc_y"] ** 2 + test["acc_z"] ** 2) ** (
    0.5
)
test["grav_acc_mag"] = (
    test["grav_acc_x"] ** 2 + test["grav_acc_y"] ** 2 + test["grav_acc_z"] ** 2
) ** (0.5)
test["body_acc_mag"] = (
    test["body_acc_x"] ** 2 + test["body_acc_y"] ** 2 + test["body_acc_z"] ** 2
) ** (0.5)
test["body_jerk_acc_mag"] = (
    test["body_jerk_acc_x"] ** 2
    + test["body_jerk_acc_y"] ** 2
    + test["body_jerk_acc_z"] ** 2
) ** (0.5)
test["gy_mag"] = (test["gy_x"] ** 2 + test["gy_y"] ** 2 + test["gy_z"] ** 2) ** (0.5)
test["body_gy_mag"] = (
    test["body_gy_x"] ** 2 + test["body_gy_y"] ** 2 + test["body_gy_z"] ** 2
) ** (0.5)
test["body_jerk_gy_mag"] = (
    test["body_jerk_gy_x"] ** 2
    + test["body_jerk_gy_y"] ** 2
    + test["body_jerk_gy_z"] ** 2
) ** (0.5)

train.to_csv(path + "train_feature_time_signal.csv", index=False)
test.to_csv(path + "test_feature_time_signal.csv", index=False)
train = pd.read_csv(path + "train_feature_time_signal.csv")
test = pd.read_csv(path + "test_feature_time_signal.csv")

features = train.keys().tolist()
features.remove("time")
print("Groupby Start")
X_train = (
    train[features]
    .groupby("id")
    .aggregate(
        [
            np.min,
            np.max,
            np.mean,
            np.std,
            np.sum,
            np.var,
            np.argmin,
            np.argmax,
            q75,
            q50,
            q25,
        ]
    )
)
X_test = (
    test[features]
    .groupby("id")
    .aggregate(
        [
            np.min,
            np.max,
            np.mean,
            np.std,
            np.sum,
            np.var,
            np.argmin,
            np.argmax,
            q75,
            q50,
            q25,
        ]
    )
)
print("Groupby End")
X_train.columns = [a + b for a, b in X_train.keys()]
X_train.reset_index(inplace=True)
y = train_y["label"]
y = y.astype(int)
X_test.columns = [a + b for a, b in X_test.keys()]
X_test.reset_index(inplace=True)

# %%
import math

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

pi = math.pi
pd.options.display.max_columns = 500
# %%
path = "../input/ai-hackaton/"
train = pd.read_csv(path + "train_features.csv")
y = pd.read_csv(path + "train_labels.csv")
test = pd.read_csv(path + "test_features.csv")
sub = pd.read_csv(path + "sample_submission.csv")
# %%
def range_func(x):
    max_val = np.max(x)
    min_val = np.min(x)
    range_val = max_val - min_val
    return range_val


def iqr_func2(x):
    q3, q1 = np.percentile(x, [20, 80])
    iqr = q3 - q1
    return iqr


def iqr_func3(x):
    q3, q1 = np.percentile(x, [40, 60])
    iqr = q3 - q1
    return iqr


def iqr_func4(x):
    q3, q1 = np.percentile(x, [15, 95])
    iqr = q3 - q1
    return iqr


def premad(x):
    return np.median(np.absolute(x - np.median(x, axis=0)), axis=0)


def preskew(x):
    return skew(x)


def prekurt(x):
    return kurtosis(x, fisher=True)


# %%
change_train = train.drop("time", axis=1)
train_change = change_train.groupby("id").diff().reset_index().fillna(0)
change_test = test.drop("time", axis=1)
test_change = change_test.groupby("id").diff().reset_index().fillna(0)
train_change.rename(columns={"index": "id"}, inplace=True)
test_change.rename(columns={"index": "id"}, inplace=True)
# %%
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

# %%
# 자이로스코프 무게중심
train["gy_Centerofgravity"] = (train["gy_x"] + train["gy_y"] + train["gy_z"]) / 3
test["gy_Centerofgravity"] = (test["gy_x"] + test["gy_y"] + test["gy_z"]) / 3

# %%
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
# %%
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

# %%
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

# %%
# 가속도계 첫번째 데이터
train_acc_head1 = train.groupby(["id"])[["acc_x", "acc_y", "acc_z"]].first()
train_acc_head1.columns = ["id", "first_acc_x", "first_acc_y", "first_acc_z"]
train_acc_head1.set_index("id", inplace=True)

test_acc_head1 = test.groupby(["id"])[["acc_x", "acc_y", "acc_z"]].first()
test_acc_head1.columns = ["id", "first_acc_x", "first_acc_y", "first_acc_z"]
test_acc_head1.set_index("id", inplace=True)

# 가속도계 첫 3초
train_acc_head = (
    train.loc[:, ["id", "time", "acc_x", "acc_y", "acc_z"]][train["time"] < 150]
    .drop("time", axis=1)
    .groupby("id")
    .mean()
)
train_acc_head.columns = ["id", "head_acc_x", "head_acc_y", "head_acc_z"]
train_acc_head = train_acc_head.groupby("id").mean()

test_acc_head = (
    test.loc[:, ["id", "time", "acc_x", "acc_y", "acc_z"]][train["time"] < 150]
    .drop("time", axis=1)
    .groupby("id")
    .mean()
)
test_acc_head.columns = ["id", "head_acc_x", "head_acc_y", "head_acc_z"]
test_acc_head = test_acc_head.groupby("id").mean()

train_preprocess = pd.concat(
    [train_preprocess, train_acc_head, train_acc_head1], axis=1
)
test_preprocess = pd.concat([test_preprocess, test_acc_head, test_acc_head1], axis=1)

# 자이로스코프 첫 3초
train_gy_head = (
    train.loc[:, ["id", "time", "gy_x", "gy_y", "gy_z"]][train["time"] < 150]
    .drop("time", axis=1)
    .groupby("id")
    .mean()
)
train_gy_head.columns = ["id", "head_gy_x", "head_gy_y", "head_gy_z"]
train_gy_head = train_gy_head.groupby("id").mean()

test_gy_head = (
    train.loc[:, ["id", "time", "gy_x", "gy_y", "gy_z"]][train["time"] < 150]
    .drop("time", axis=1)
    .groupby("id")
    .mean()
)
test_gy_head.columns = ["id", "head_gy_x", "head_gy_y", "head_gy_z"]
test_gy_head = test_gy_head.groupby("id").mean()

train_preprocess = pd.concat([train_preprocess, train_gy_head], axis=1)
test_preprocess = pd.concat([test_preprocess, test_gy_head], axis=1)
# %%
train.groupby(["id"])[["acc_x", "acc_y", "acc_z"]].first().reset_index()
# %%
# 가속도계 첫번째 데이터
train_acc_head1 = train.groupby(["id"])[["acc_x", "acc_y", "acc_z"]].first().reset_index()
train_acc_head1.columns = ["id", "first_acc_x", "first_acc_y", "first_acc_z"]
train_acc_head1.set_index("id", inplace=True)

test_acc_head1 = test.groupby(["id"])[["acc_x", "acc_y", "acc_z"]].first().reset_index()
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
test_preprocess = pd.concat([test_preprocess, test_acc_head, test_acc_head1], axis=1)

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
    train.loc[:, ["id", "time", "gy_x", "gy_y", "gy_z"]][train["time"] < 150]
    .drop("time", axis=1)
    .groupby("id")
    .mean()
    .reset_index()
)
test_gy_head.columns = ["id", "head_gy_x", "head_gy_y", "head_gy_z"]
test_gy_head = test_gy_head.groupby("id").mean()

train_preprocess = pd.concat([train_preprocess, train_gy_head], axis=1)
test_preprocess = pd.concat([test_preprocess, test_gy_head], axis=1)
# %%

# %%
train_preprocess
# %%
