"""Every suite root must state the no-decay policy rather than inherit it silently.

Hydra replaces ``no_weight_decay`` instead of appending, so a root that omits it tracks
``DEFAULT_NO_WEIGHT_DECAY`` while the roots that state it do not, and the four drift apart
the moment the code default changes.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from core.optim import DEFAULT_NO_WEIGHT_DECAY

SRC = Path(__file__).parents[1]
PACKAGES = ("ts1", "ts2", "ts3", "pretrain")


@pytest.mark.parametrize("pkg", PACKAGES)
def test_every_suite_root_states_the_no_decay_policy(pkg):
    names = OmegaConf.load(SRC / pkg / "configs" / "train.yaml").get("no_weight_decay")
    assert names is not None, f"{pkg} inherits the code default instead of stating it"
    missing = set(DEFAULT_NO_WEIGHT_DECAY) - set(names)
    assert not missing, f"{pkg} drops {sorted(missing)} from the default"
