import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@hydra.main(config_path="../../config/train/", config_name="ensemble.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission = pd.read_csv(path + "sample_submission.csv")

    cnn_fold = pd.read_csv(submit_path + "10fold_cnn.csv")
    cat_depth = pd.read_csv(submit_path + "catboost_depth4.csv")
    submission.iloc[:, 1:] = (
        cfg.weight.w1 * cnn_fold.iloc[:, 1:] + cfg.weight.w2 * cat_depth.iloc[:, 1:]
    )

    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
