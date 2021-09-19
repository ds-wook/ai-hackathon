from functools import partial

import hydra
import pandas as pd
from omegaconf import DictConfig
from tuning.bayesian import BayesianOptimizer, xgb_objective


@hydra.main(config_path="../../config/optimization/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = hydra.utils.to_absolute_path(cfg.dataset.path) + "/"
    X = pd.read_csv(path + cfg.dataset.train)
    train_y = pd.read_csv(path + "train_labels.csv")
    y = train_y["label"]

    objective = partial(xgb_objective, X=X, y=y, n_fold=cfg.model.fold)
    bayesian_optim = BayesianOptimizer(objective)
    study = bayesian_optim.build_study(trials=cfg.optimization.trials)
    bayesian_optim.xgb_save_params(study, cfg.optimization.params)


if __name__ == "__main__":
    _main()
