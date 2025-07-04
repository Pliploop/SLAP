from math import cos, pi
from typing import Sequence

import torch
from lightning.pytorch import Callback, LightningModule, Trainer


class MAWeightUpdate(Callback):
    """Weight update rule from BYOL.
    
    Code stolen and adapted from Lightning Bolts
    """

    def __init__(self,
                 initial_tau: float = 0.996,
                 final_tau: float | int = 1.,
                 every_n_steps: int = 1,
                 update_method: str = "cos"):
        """
        Args:
            initial_tau: starting tau. Auto-updates with every training step
        """
        super().__init__()
        self.initial_tau = initial_tau
        self.final_tau = final_tau
        self.every_n_steps = every_n_steps

        self.current_tau = initial_tau

        if update_method == "cos":
            self.update_tau = self.update_tau_cos
        elif update_method == "exp":
            self.update_tau = self.update_tau_exp
        elif update_method == "lin":
            self.update_tau = self.update_tau_lin
        else:
            raise ValueError(f"Unknown update method {update_method}")

    def on_train_batch_end(
            self,
            trainer: Trainer,
            pl_module: LightningModule,
            outputs: Sequence,
            batch: Sequence,
            batch_idx: int
    ) -> None:
       # update weights every n steps
        if batch_idx % self.every_n_steps != 0:
            return

        self.update_weights(pl_module.audio_encoder,
                            pl_module.audio_ema)
        self.update_weights(pl_module.text_encoder,
                            pl_module.text_ema)

        # log tau
        pl_module.log("MA rate", self.current_tau, prog_bar=False, logger=True)

        # update tau after
        self.current_tau = self.update_tau(pl_module, trainer)

    def update_tau_cos(self, pl_module: LightningModule, trainer: Trainer) -> float:
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs
        tau = self.final_tau - (self.final_tau - self.initial_tau) * (cos(pi * pl_module.global_step / max_steps) + 1) / 2
        return tau
    
    def update_tau_exp(self, pl_module: LightningModule, trainer: Trainer) -> float:
        half_life = len(trainer.train_dataloader) * self.final_tau
        return 1 - self.initial_tau * 2 ** (- trainer.global_step / half_life)

    def update_tau_lin(self, pl_module: LightningModule, trainer: Trainer) -> float:
        max_steps = len(trainer.train_dataloader) * trainer.max_epochs
        tau = self.initial_tau + (self.final_tau - self.initial_tau) * pl_module.global_step / max_steps
        return tau

    def update_weights(
        self,
        student_network: torch.nn.Module,
        teacher_network: torch.nn.Module
    ) -> None:
        # apply MA weight update
        for (name, student_p), (_, teacher_p) in zip(
            student_network.named_parameters(),
            teacher_network.named_parameters(),
        ):
            teacher_p.data = self.current_tau * teacher_p.data + (1 - self.current_tau) * student_p.data