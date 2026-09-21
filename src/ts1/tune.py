import hydra
from omegaconf import DictConfig

from core.launch import run
from core.tune import tune_suite


@hydra.main(version_base="1.3", config_path="./configs", config_name="tune.yaml")
def main(cfg: DictConfig):
    tune_suite(cfg)


if __name__ == "__main__":
    run(main)
