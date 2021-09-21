import hydra
import numpy as np
import pandas as pd
from data.features import select_features
from hydra.utils import to_absolute_path
from model.tree import train_kfold_ext
from omegaconf import DictConfig


@hydra.main(config_path="../../config/train/", config_name="extra.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train_y = pd.read_csv(path + "train_labels.csv")
    X = pd.read_csv(path + cfg.dataset.train)
    X_test = pd.read_csv(path + cfg.dataset.test)
    y = train_y["label"]

    # select feature
    X, X_test = select_features(X, y, X_test)

    # model
    ext_oof, ext_preds = train_kfold_ext(cfg.model.fold, X, y, X_test)
    submission = pd.read_csv(path + "sample_submission.csv")
    submission.iloc[:, 1:] = ext_preds
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    np.save(submit_path + "ext_oof.npy", ext_oof)
    np.save(submit_path + "ext_pred.npy", ext_preds)
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
