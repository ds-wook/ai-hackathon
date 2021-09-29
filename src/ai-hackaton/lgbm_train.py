import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from model.boosting_tree import train_kfold_lightgbm
from omegaconf import DictConfig


@hydra.main(config_path="../../config/train/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    train_y = pd.read_csv(path + "train_labels.csv")
    X = pd.read_csv(path + cfg.dataset.train)
    X_test = pd.read_csv(path + cfg.dataset.test)
    X["is_exercise"] = np.where(train_y["label"] == 26, 0, 1)
    y = X["is_exercise"]

    # model
    lgbm_oof, lgbm_preds = train_kfold_lightgbm(
        cfg.model.fold, X.drop(["is_exercise"], axis=1), y, X_test, dict(cfg.params), cfg.model.verbose
    )

    X_test["is_exercise"] = np.where(lgbm_preds < 0.5, 0, 1)
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    X.to_csv(submit_path + "final_X.csv", index=False)
    X_test.to_csv(submit_path + "final_X_test.csv", index=False)


if __name__ == "__main__":
    _main()
