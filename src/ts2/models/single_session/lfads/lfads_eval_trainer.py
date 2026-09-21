import torch

from core.utils.exceptions import TrainingConstraintsError
from ts2.ts2_eval_trainer import TS2EvalTrainer


class LFADSEvalTrainer(TS2EvalTrainer):
    """TS2 trainer for LFADS.

    Replaces the base reconstruction loss with the LFADS ELBO:

        L = recon + l2_ramp * (l2_gen_scale * l2_gen + l2_con_scale * l2_con)
                  + kl_ramp * (kl_ic_scale * kl_ic + kl_co_scale * kl_co)

    All three groups are per-sample sums averaged over the batch, which makes the
    KL scales true ELBO weights (1.0 is the unmodified ELBO). Weighting a
    per-element mean reconstruction against per-sample summed KL terms is what
    produces posterior collapse.
    """

    def training_step(self, X: dict, y: dict) -> torch.Tensor:
        kl_ramp = self._ramp(self.cfg.kl_start_epoch, self.cfg.kl_increase_epoch)
        l2_ramp = self._ramp(self.cfg.l2_start_epoch, self.cfg.l2_increase_epoch)

        try:
            out = self.model(**X["model_inputs"], return_posterior=True)
        except ValueError as e:
            # A non-finite posterior trips torch.distributions' constraint
            # check; report it so the tuner prunes the trial.
            raise TrainingConstraintsError(
                f"LFADS diverged (posterior became non-finite): {e}"
            ) from e

        recon = self._reconstruction(out, y["values"])
        kl = self.cfg.kl_ic_scale * out["kl_ic"] + self.cfg.kl_co_scale * out["kl_co"]
        l2 = self.cfg.l2_gen_scale * out["l2_gen"] + self.cfg.l2_con_scale * out["l2_con"]
        loss = recon + kl_ramp * kl + l2_ramp * l2

        if not torch.isfinite(loss):
            raise TrainingConstraintsError(f"LFADS diverged: non-finite loss ({loss.item()})")

        if self.cfg.log_train_step:
            self.logger.log("train/recon", recon.item())
            self.logger.log("train/kl_ic", out["kl_ic"].item())
            self.logger.log("train/kl_co", out["kl_co"].item())
            self.logger.log("train/l2_gen", out["l2_gen"].item())
            self.logger.log("train/l2_con", out["l2_con"].item())
            self.logger.log("train/kl_ramp", kl_ramp)
            self.logger.log("train/l2_ramp", l2_ramp)

        return loss

    def _reconstruction(self, out: dict, target: torch.Tensor) -> torch.Tensor:
        """Poisson NLL summed within a sample, averaged over the batch.

        Where the model hid part of its input, gradient is blocked on the entries
        the encoder saw, so it cannot score by copying. The reported value still
        covers every entry.
        """
        nll = self.loss_fn(out["log_rates"], target)

        grad_mask = out.get("grad_mask")
        if grad_mask is not None:
            nll = nll * grad_mask + (nll * (1 - grad_mask)).detach()

        return nll.sum(dim=(-2, -1)).mean()

    def _ramp(self, start_epoch: int, increase_epochs: int) -> float:
        """Coefficient ramping linearly from 0 to 1 over ``increase_epochs``."""
        if increase_epochs <= 0:
            return 1.0 if self.epoch >= start_epoch else 0.0
        return min(max((self.epoch - start_epoch) / increase_epochs, 0.0), 1.0)
