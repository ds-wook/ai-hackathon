from functools import partial

import hydra
import numpy as np
import pandas as pd
from data.features import select_features
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tuning.bayesian import BayesianOptimizer, lgbm_objective


@hydra.main(config_path="../../config/optimization/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = hydra.utils.to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"
    train_y = pd.read_csv(path + "train_labels.csv")
    cnn_oof = np.load(submit_path + "10fold_cnn.npy")
    cnn_preds = pd.read_csv(submit_path + "10fold_cnn.csv")

    features = [f"prob_{i}" for i in range(61)]

    X = pd.DataFrame(cnn_oof, columns=features)
    X_test = cnn_preds.drop(["id"], axis=1)
    y = train_y["label"]

    # select features
    X, X_test = select_features(X, y, X_test)

    objective = partial(lgbm_objective, X=X, y=y, n_fold=cfg.model.fold)
    bayesian_optim = BayesianOptimizer(objective)
    study = bayesian_optim.build_study(trials=cfg.optimization.trials)
    bayesian_optim.lgbm_save_params(study, cfg.optimization.params)


if __name__ == "__main__":
    _main()
