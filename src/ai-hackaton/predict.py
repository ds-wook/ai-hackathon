import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@hydra.main(config_path="../../config/train/", config_name="lgbm.yaml")
def _main(cfg: DictConfig):
    path = to_absolute_path(cfg.dataset.path) + "/"
    submit_path = to_absolute_path(cfg.submit.path) + "/"

    model1 = pd.read_csv(submit_path + cfg.submit.model1)
    model2 = pd.read_csv(submit_path + cfg.submit.model2)
    submission = pd.read_csv(path + "sample_submission.csv")
    submission.iloc[:, 1:] = model1.iloc[:, 1:] * 0.8 + model2.iloc[:, 1:] * 0.2
    
    submission.to_csv(submit_path + cfg.submit.name, index=False)


if __name__ == "__main__":
    _main()
