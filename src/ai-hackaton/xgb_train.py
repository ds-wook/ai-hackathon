import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from model.boosting_tree import train_kfold_xgb
from omegaconf import DictConfig


@hydra.main(config_path="../../config/train/", config_name="xgb.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    train_y = pd.read_csv(path + "train_labels.csv")
    cnn_oof = np.load(submit_path + "10fold_cnn.npy")
    cnn_preds = pd.read_csv(submit_path + "10fold_cnn.csv")

    features = [f"prob_{i}" for i in range(61)]

    X = pd.DataFrame(cnn_oof, columns=features)
    X["movement_preds"] = np.argmax(cnn_oof, axis=1)
    X_test = cnn_preds.drop(["id"], axis=1)
    X_test["movement_preds"] = np.argmax(X_test.values.reshape(-1, 61), axis=1)
    y = train_y["label"]

    # model
    xgb_oof, xgb_preds = train_kfold_xgb(
        cfg.model.fold, X, y, X_test, dict(cfg.params), cfg.model.verbose
    )

    submission = pd.read_csv(path + "sample_submission.csv")
    submission.iloc[:, 1:] = xgb_preds

    np.save(submit_path + "xgb_oof.npy", xgb_oof)
    np.save(submit_path + "xgb_pred.npy", xgb_preds)
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
