"""Train one TS2 config over every eval seed and report mean ± std ± sem.

A shim like train.py and tune.py: the runner is core.eval_seeds.run_seeds, and this
file exists only so Hydra resolves config_path against this package's configs/.
"""

import hydra
from omegaconf import DictConfig

from core.eval_seeds import run_seeds
from core.launch import run


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    run_seeds(cfg)


if __name__ == "__main__":
    run(main)
