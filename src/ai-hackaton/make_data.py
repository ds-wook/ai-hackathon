import pandas as pd

path = "../input/ai-hackaton/"
agg_train_x = pd.read_csv(path + "agg_train.csv")
agg_test_x = pd.read_csv(path + "agg_test.csv")

log_change_train = pd.read_csv(path + "log_change_train.csv")
log_change_test = pd.read_csv(path + "log_change_test.csv")

log_skip_train = pd.read_csv(path + "log_skip_train.csv")
log_skip_test = pd.read_csv(path + "log_skip_test.csv")

smoothed_agg_train = pd.read_csv(path + "smoothed_agg_train.csv")
smoothed_agg_test = pd.read_csv(path + "smoothed_agg_test.csv")

count_stop_moving_train = pd.read_csv(path + "count_stop_moving_train.csv")
count_stop_moving_test = pd.read_csv(path + "count_stop_moving_test.csv")

fft_train = pd.read_csv(path + "fft_train.csv")
fft_test = pd.read_csv(path + "fft_test.csv")

total_train_x = pd.concat(
    [
        agg_train_x,
        log_change_train,
        log_skip_train,
        smoothed_agg_train,
        count_stop_moving_train,
        fft_train,
    ],
    axis=1,
).drop(columns=["id"])
total_test_x = pd.concat(
    [
        agg_test_x,
        log_change_test,
        log_skip_test,
        smoothed_agg_test,
        count_stop_moving_test,
        fft_test,
    ],
    axis=1,
).drop(columns=["id"])

print(total_train_x.shape)
print(total_test_x.shape)

total_train_x.to_csv(path + "concated_train_x.csv", index=False)
total_test_x.to_csv(path + "concated_test_x.csv", index=False)
