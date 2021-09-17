import hydra
import pandas as pd
from data.dataset import load_dataset
from hydra.utils import to_absolute_path
from model.boosting_tree import train_kfold_lightgbm
from omegaconf import DictConfig


@hydra.main(config_path="../../config/train/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    train, test = load_dataset(path)
    X = train.drop("label", axis=1)
    y = train["label"]
    lgbm_preds = train_kfold_lightgbm(
        cfg.model.fold, X, y, test, dict(cfg.params), cfg.model.verbose
    )

    submission = pd.read_csv(path + "sample_submission.csv")
    submission.iloc[:, 1:] = lgbm_preds
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
