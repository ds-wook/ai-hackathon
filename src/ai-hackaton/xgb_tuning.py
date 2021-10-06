from functools import partial

import hydra
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tuning.bayesian import BayesianOptimizer, xgb_objective


@hydra.main(config_path="../../config/optimization/", config_name="xgb.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    train_y = pd.read_csv(path + "train_labels.csv")

    cnn_oof = np.load(submit_path + "10fold_cnn_oof.npy")
    cat_oof = np.load(submit_path + "10fold_cat_oof.npy")
    auto_oof = np.load(submit_path + "10fold_automl_oof.npy")

    features = [f"{c}_prob_{i}" for c in ["cnn", "cat", "automl"] for i in range(61)]
    X = np.concatenate([cnn_oof, cat_oof, auto_oof], axis=1)
    X = pd.DataFrame(X, columns=features)

    y = train_y["label"]

    objective = partial(xgb_objective, X=X, y=y, n_fold=cfg.model.fold)
    bayesian_optim = BayesianOptimizer(objective)
    study = bayesian_optim.build_study(trials=cfg.optimization.trials)
    bayesian_optim.xgb_save_params(study, cfg.optimization.params)


if __name__ == "__main__":
    _main()
