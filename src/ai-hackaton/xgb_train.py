import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.metrics import log_loss
from trainer.boosting_tree import XGBTrainer


@hydra.main(config_path="../../config/train/", config_name="xgb.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    train_y = pd.read_csv(path + "train_labels.csv")

    cnn_oof = np.load(submit_path + "10fold_cnn_oof.npy")
    cnn_preds = np.load(submit_path + "10fold_cnn_preds.npy")
    cat_oof = np.load(submit_path + "10fold_cat_oof.npy")
    cat_preds = np.load(submit_path + "10fold_cat_preds.npy")
    auto_oof = np.load(submit_path + "10fold_automl_oof.npy")
    auto_preds = np.load(submit_path + "10fold_automl_preds.npy")

    features = [f"{c}_prob_{i}" for c in ["cnn", "cat", "automl"] for i in range(61)]
    X = np.concatenate([cnn_oof, cat_oof, auto_oof], axis=1)
    X = pd.DataFrame(X, columns=features)

    X_test = np.concatenate([cnn_preds, cat_preds, auto_preds], axis=1)
    X_test = pd.DataFrame(X_test, columns=features)
    y = train_y["label"]

    # model
    xgb_trainer = XGBTrainer(cfg.model.fold, log_loss)
    xgb_trainer.train(X, y, cfg.params, cfg.model.verbose)
    xgb_preds = xgb_trainer.predict(X_test)

    submission = pd.read_csv(path + "sample_submission.csv")
    submission.iloc[:, 1:] = xgb_preds
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
