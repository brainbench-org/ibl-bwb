import hydra
from omegaconf import DictConfig

from core.launch import launch, run


@hydra.main(version_base="1.2", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    launch(cfg)


if __name__ == "__main__":
    run(main)
