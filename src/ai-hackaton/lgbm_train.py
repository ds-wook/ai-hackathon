import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from model.boosting_tree import train_kfold_logistic
from omegaconf import DictConfig


@hydra.main(config_path="../../config/train/", config_name="logistic.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    train_y = pd.read_csv(path + "train_labels.csv")
    X = pd.read_csv(path + cfg.dataset.train)
    X_test = pd.read_csv(path + cfg.dataset.test)
    submission = pd.read_csv(path + "sample_submission.csv")
    y = train_y.label

    # model
    lr_oof, lr_preds = train_kfold_logistic(
        cfg.model.fold,
        X,
        y,
        X_test,
        dict(cfg.params),
        cfg.model.verbose,
    )

    submission.iloc[:, 1:] = lr_preds
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission.to_csv(submit_path + "10fold_logistic.csv", index=False)


if __name__ == "__main__":
    _main()
