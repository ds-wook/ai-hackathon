import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@hydra.main(config_path="../../config/train/", config_name="ensemble.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    submission = pd.read_csv(path + "sample_submission.csv")

    cnn_preds = pd.read_csv(submit_path + "10fold_cnn.csv")
    cat_preds = pd.read_csv(submit_path + "catboost_depth4.csv")
    xgb_preds = pd.read_csv(submit_path + "xgb_stacking_cnn.csv")

    submission.iloc[:, 1:] = (
        cfg.weight.w1 * cnn_preds.iloc[:, 1:]
        + cfg.weight.w2 * cat_preds.iloc[:, 1:]
        + cfg.weight.w3 * xgb_preds.iloc[:, 1:]
    )

    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
