"""Tests for the search path :class:`ModelConfigDiscovery` builds.

The plugin decides, per entry point, which model packages ship configs into the run. It
is the only thing standing between an app and a ``trainer=`` name it should never see,
and getting it wrong is quiet: a suite that picks up a foreign package composes a config
that resolves, instantiates, and trains the wrong thing. The contract is pinned as the
exact reachable set, read off the filesystem, so moving a model package fails here first.
"""

from pathlib import Path

import pytest
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

SRC = Path(__file__).parents[1]

# Which model packages each app is meant to reach: its own, plus pretrain's when it
# finetunes (ts1, ts2) or calibrates (ts3's transductive trainers) and so composes
# pretrain's model group. A pretrain run sits in no suite.
REACHES = {
    "ts1": ["ts1/models", "pretrain/models"],
    "ts2": ["ts2/models", "pretrain/models"],
    "ts3": ["ts3/models", "pretrain/models"],
    "pretrain": ["pretrain/models"],
}
GROUPS = ("trainer", "model", "extractor")


def _files(app: str, group: str) -> list[Path]:
    """Every config the app's own configs/ and its declared model packages define."""
    dirs = [SRC / app / "configs" / group]
    dirs += [d for pkg in REACHES[app] for d in (SRC / pkg).rglob(f"configs/{group}")]
    return [f for d in dirs for f in d.glob("*.yaml")]


def _declared(app: str) -> set[str]:
    return {f.stem for f in _files(app, "trainer")}


def _reachable(app: str) -> set[str]:
    with initialize_config_dir(config_dir=str(SRC / app / "configs"), version_base="1.2"):
        return set(GlobalHydra.instance().config_loader().get_group_options("trainer"))


@pytest.mark.parametrize("app", list(REACHES))
def test_an_app_reaches_exactly_the_packages_it_declares(app):
    assert _reachable(app) == _declared(app)


@pytest.mark.parametrize("app", list(REACHES))
@pytest.mark.parametrize("group", GROUPS)
def test_no_two_packages_claim_one_config_name(app, group):
    """Two packages defining one name is invisible in the reachable set: hydra serves the
    first on the search path and the other never loads, so assert on the files instead."""
    files = _files(app, group)
    dupes = {f.stem for f in files if sum(g.stem == f.stem for g in files) > 1}
    assert not dupes, f"{app} reaches {len(dupes)} shadowed {group} name(s): {sorted(dupes)}"
