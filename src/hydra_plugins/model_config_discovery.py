import importlib
import importlib.util
from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

SUITES = ("ts1", "ts2", "ts3")


def _package_dir(name: str) -> Path:
    """Where a package lives on disk, located without importing it."""
    return Path(importlib.util.find_spec(name).origin).parent.resolve()


class ModelConfigDiscovery(SearchPathPlugin):
    """Put every ``<pkg>/models/**/configs`` on the Hydra search path.

    A model then ships its own ``model/``, ``trainer/`` and ``extractor/`` groups without
    touching central config files.

    An app (an entry point under ``@hydra.main``, such as ``ts1/train.py``, whose
    ``config_path`` Hydra registers under the provider ``"main"``) reaches a package
    exactly when it composes a group from it: always its own ``<app>/models/``, plus
    ``pretrain/models/`` only to finetune or calibrate, because every such trainer opens
    with ``override /model: <a pretrain model>``. TS3's inductive and supervised extractors
    compose no group at all, since they rebuild the model from the config saved inside the
    checkpoint they read, but its transductive calibration trainers do, the same way
    TS1/TS2 finetuning does.
    """

    provider = "model-configs"

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Which suite is running: the app's own config dir, which hydra tags "main", sits
        # inside exactly one suite package. Tested by path containment, not by matching the
        # name in a string, so a checkout directory called ts3 cannot claim the suite.
        primary = next(
            (e.path for e in search_path.config_search_path if e.provider == "main"), None
        )
        app_dir = Path(primary.removeprefix("file://")).resolve() if primary else None
        suites = [name for name in SUITES if app_dir and app_dir.is_relative_to(_package_dir(name))]
        model_pkgs = [importlib.import_module(f"{name}.models") for name in suites]

        # A models/pretrained/ or models/transductive/ is the proxy for "finetunes or
        # calibrates from a pretrain checkpoint". A pretrain run sits in no suite: reaching
        # the suites too would put ts1's and ts2's same-named finetune trainers on one
        # search path, where the second is silently unreachable.
        if not model_pkgs or any(
            (Path(pkg.__file__).parent / sub).is_dir()
            for pkg in model_pkgs
            for sub in ("pretrained", "transductive")
        ):
            model_pkgs.insert(0, importlib.import_module("pretrain.models"))

        for model_pkg in model_pkgs:
            model_dir = Path(model_pkg.__file__).parent
            for configs_dir in sorted(model_dir.rglob("configs")):
                if configs_dir.is_dir():
                    search_path.append(
                        provider=f"model-{configs_dir.parent.relative_to(model_dir)}",
                        path=f"file://{configs_dir}",
                    )
