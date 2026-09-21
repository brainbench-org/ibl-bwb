"""Shared Ray Tune driver.

Each suite keeps its own ``tune.py`` for one reason: Hydra resolves ``config_path``
relative to the decorated function's file, so the shim must sit in the package whose
``configs/`` it wants. The shims are otherwise identical.
"""

import datetime
import os

import optuna
import torch

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import copy
import traceback
from functools import partial

import hydra
import ray
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from ray import tune
from ray.tune.search.optuna.optuna_search import OptunaSearch

from core.eval_seeds import run_final_eval
from core.launch import train
from core.model import BaseModel
from core.utils.exceptions import TrainingConstraintsError
from core.utils.logger import get_cli_logger

logger = get_cli_logger()


def _fill_config(tune_params: dict, cfg: DictConfig, process_params: bool = True) -> DictConfig:
    # process tunable parameters for the model
    model_cls = hydra.utils.get_class(cfg.model._target_)
    if process_params and model_cls is not None and issubclass(model_cls, BaseModel):
        tune_params = model_cls.process_tunable_params(tune_params)

    # fill config with tuned parameters
    config = copy.deepcopy(cfg)
    for key, value in tune_params.items():
        OmegaConf.update(config, key, value)
    return config


def trainable(tune_params: dict, cfg: DictConfig, seed: int):
    try:
        cfg = _fill_config(tune_params, cfg)

        # log resolved config to trial directory
        trial_dir = tune.get_context().get_trial_dir()
        with open(os.path.join(trial_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        with open_dict(cfg):
            cfg.seed = seed
            # Predictions are only wanted from the final multi-seed eval, not every
            # search trial.
            cfg.save_preds.enable = False

        # Ray parallelizes across trials, not inside one: a trial is sized for a
        # fraction of a GPU (ray.gpu), so it runs as a single-process world.
        results = train(cfg, 0, 1)

        tune.report(
            {
                # reporting best metrics first so they show up in the running report
                "best/val/avg": results["best/val/avg"],
                "best/test/avg": results["best/test/avg"],
                **results,
            }
        )
    except TrainingConstraintsError as e:
        logger.warning(f"Invalid Hyperparameters: {e}")
        logger.error(traceback.format_exc())
        raise optuna.TrialPruned() from None
    except torch.cuda.OutOfMemoryError as e:
        logger.warning(f"Trial crashed with CUDA OOM: {e}")
        logger.error(traceback.format_exc())
        torch.cuda.empty_cache()
        raise optuna.TrialPruned() from None
    except Exception as e:
        logger.error(f"Trial crashed with error: {e}")
        logger.error(traceback.format_exc())
        raise e


def tune_suite(cfg: DictConfig):
    hydra_choices = HydraConfig.get().runtime.choices
    with open_dict(cfg):
        cfg.hydra_choices = hydra_choices

    ray_address = os.getenv("RAY_ADDRESS")
    ray_init_kwargs = {"address": ray_address, "ignore_reinit_error": True}
    if not ray_address:
        ray_init_kwargs["num_cpus"] = len(os.sched_getaffinity(0))
        ray_init_kwargs["num_gpus"] = torch.cuda.device_count()
    ray.init(**ray_init_kwargs)
    logger.info(f"RAY_ADDRESS={ray_address}")

    n_gpu = ray.cluster_resources().get("GPU", 0)
    if n_gpu < cfg.ray.gpu:
        raise RuntimeError(
            f"ray sees {n_gpu} GPUs, each trial needs {cfg.ray.gpu}; tune would idle until walltime"
        )
    logger.info(f"ray resources: {ray.cluster_resources()}")

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    assert isinstance(cfg.recording_id, str) or len(cfg.recording_id) == 1, (
        "recording_id must be a string or a list with one element"
    )

    run_name = (
        f"{slurm_job_id}_" if slurm_job_id else ""
    ) + f"{cfg.wandb.project}_{cfg.recording_id}_{cfg.task}_{timestamp}"

    def trial_name_creator(trial):
        return (
            (f"{slurm_job_id}_" if slurm_job_id else "")
            + f"{cfg.wandb.project}_{cfg.recording_id}_{cfg.task}_{trial.trial_id}_{datetime.datetime.now().strftime('%m%d_%H%M%S')}"
        )

    # adjust wandb parameters for the sweep
    if slurm_job_id:
        cfg.wandb.dir = os.path.join(cfg.wandb.dir, f"{slurm_job_id}_{timestamp}")
        os.makedirs(cfg.wandb.dir, exist_ok=True)
        logger.info(f"Wandb directory set to: {cfg.wandb.dir}")

    # create the output directories
    os.makedirs(cfg.ray.results_dir, exist_ok=True)
    os.makedirs(cfg.ray.outputs_dir, exist_ok=True)

    # initialize tuner
    model_cls = hydra.utils.get_class(cfg.model._target_)
    algo = OptunaSearch(
        space=partial(model_cls.create_search_space, cfg=cfg),
        metric="best/val/avg",
        mode="max",
        seed=cfg.ray.search_seed,
    )
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                trainable=trainable,
                cfg=cfg,
                seed=cfg.ray.search_seed,
            ),
            resources={
                "gpu": cfg.ray.gpu,
                "cpu": cfg.ray.cpu,
            },
        ),
        tune_config=tune.TuneConfig(
            trial_name_creator=trial_name_creator,
            num_samples=cfg.ray.num_samples,
            search_alg=algo,
            max_concurrent_trials=cfg.ray.max_concurrent_trials,
        ),
        run_config=tune.RunConfig(
            name=run_name,
            storage_path=os.path.abspath(cfg.ray.results_dir),
            failure_config=tune.FailureConfig(max_failures=0),
        ),
    )

    # run the sweep
    results = tuner.fit()

    # summarize best results
    best_result = results.get_best_result(metric="best/val/avg", mode="max")
    logger.info(f"==== TUNE COMPLETED: {cfg.task} ====")
    logger.info(f"Best Val Metric: {best_result.metrics['best/val/avg']:.4f}")
    logger.info(f"Best Test Metric: {best_result.metrics['best/test/avg']:.4f}")
    for k, v in best_result.metrics.items():
        if k.startswith("test/"):
            logger.info(f"Test {k.split('/')[1]}: {v:.4f}")
    logger.info(f"Best Config: {best_result.config}")

    # save results to csv
    df = results.get_dataframe()
    sorted_df = df.sort_values(by="best/val/avg", ascending=False)
    output_csv = os.path.join(os.path.abspath(cfg.ray.outputs_dir), f"{run_name}_ray_results.csv")
    sorted_df.to_csv(output_csv, index=False)
    logger.info(f"Saved Ray Tune results to: {output_csv}")

    # run final results on multiple seeds with best config
    logger.info("\n")
    cfg = _fill_config(best_result.config, cfg, process_params=False)
    if hasattr(cfg, "eval_seeds"):
        df = run_final_eval(cfg, display_results=True)

        # save final results to csv
        df.to_csv(
            os.path.join(cfg.ray.outputs_dir, f"{run_name}_final_results.csv"),
            index=False,
        )
        logger.info(
            f"Saved Final Results to: {os.path.join(cfg.ray.outputs_dir, f'{run_name}_final_results.csv')}"
        )
