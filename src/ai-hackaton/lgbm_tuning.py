from functools import partial

import hydra
import pandas as pd
from data.dataset import add_tau_feature, create_agg_features
from omegaconf import DictConfig
from tuning.bayesian import BayesianOptimizer, lgbm_objective


@hydra.main(config_path="../../config/optimization/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = hydra.utils.to_absolute_path(cfg.dataset.path) + "/"
    train = pd.read_pickle(path + cfg.dataset.train)
    test = pd.read_pickle(path + cfg.dataset.test)

    train, test = add_tau_feature(train, test)
    train, test = create_agg_features(train, test, path)
    print(train.shape, test.shape)
    # Split features and target
    X = train.drop(["row_id", "target", "time_id"], axis=1)
    y = train["target"]

    objective = partial(lgbm_objective, X=X, y=y, n_fold=cfg.model.fold)
    bayesian_optim = BayesianOptimizer(objective)
    study = bayesian_optim.build_study(trials=cfg.optimization.trials)
    bayesian_optim.lgbm_save_params(study, cfg.optimization.params)


if __name__ == "__main__":
    _main()
