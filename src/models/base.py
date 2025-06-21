import abc
import logging
from copy import deepcopy
from typing import Tuple, Type, Dict, Any

import torch
import torch.nn as nn

import lightning.pytorch as pl

from src.callbacks.ma_update import MAWeightUpdate

log = logging.getLogger(__name__)


class BaseModule(pl.LightningModule, metaclass=abc.ABCMeta):
    def __init__(
            self,
            audio_encoder: nn.Module,
            text_encoder: nn.Module,
            loss_fn: nn.Module,
            optimizer: Type[torch.optim.Optimizer],
            scheduler: Type[torch.optim.lr_scheduler.LRScheduler] | None = None,
            compile: bool | str = False,
            **kwargs
    ):
        super(BaseModule, self).__init__()
        self.save_hyperparameters(ignore=["audio_encoder", "text_encoder", "loss_fn", "optimizer", "scheduler"])
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer
        self.scheduler_cls = scheduler

    def on_fit_start(self) -> None:
        if self.trainer.datamodule.tokenizer.tokenizer is None:
            self.trainer.datamodule.tokenizer.tokenizer = self.text_encoder.encoder.tokenizer
            self.trainer.datamodule.tokenizer.update_options(self.trainer.datamodule.tokenizer.tokenizer_options)
            
            log.info(f'Set tokenizer to {self.text_encoder.encoder.tokenizer} as no tokenizer was set in the datamodule')

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        parameters = list(self.audio_encoder.parameters()) + list(self.text_encoder.parameters()) + list(self.loss_fn.parameters())
        optimizer = self.optimizer_cls(params=parameters)
        if self.scheduler_cls is not None:
            try:
                scheduler = self.scheduler_cls(optimizer=optimizer)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {  # TODO: adapt this to LinearWarmup thingy
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            except TypeError:
                scheduler = self.scheduler_cls(optimizer=optimizer, trainer=self.trainer)

                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": 1
                    }
                }
        return {"optimizer": optimizer}

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.audio_encoder = torch.compile(self.audio_encoder)
            self.text_encoder = torch.compile(self.text_encoder)

    def save_initial_model(self):
        self.trainer.save_checkpoint(self.trainer.checkpoint_callback.format_checkpoint_name(dict(epoch=0, step=0)))


class BaseEMAModule(BaseModule):
    def __init__(
            self,
            audio_encoder: nn.Module,
            text_encoder: nn.Module,
            loss_fn: nn.Module,
            optimizer: Type[torch.optim.Optimizer],
            scheduler: Type[torch.optim.lr_scheduler.LRScheduler] | None = None,
            ma_callback: MAWeightUpdate = MAWeightUpdate(),
            compile: bool | str = False,
            **kwargs
    ):
        super(BaseEMAModule, self).__init__(
            audio_encoder=audio_encoder,
            text_encoder=text_encoder,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            compile=compile,
            **kwargs
        )

        self.audio_ema = deepcopy(self.audio_encoder)
        for param in self.audio_ema.parameters():
            param.requires_grad = False
        
        self.text_ema = deepcopy(self.text_encoder)
        for param in self.text_ema.parameters():
            param.requires_grad = False
        
        self.ma_callback = ma_callback

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            mode = self.hparams.compile if isinstance(self.hparams.compile, str) else "default"
            self.audio_encoder = torch.compile(self.audio_encoder, mode=mode)
            self.text_encoder = torch.compile(self.text_encoder, mode=mode)

            self.audio_ema = torch.compile(self.audio_ema, mode=mode)
            self.text_ema = torch.compile(self.text_ema, mode=mode)

    def on_fit_start(self) -> None:
        super(BaseEMAModule, self).on_fit_start()#
    
    def on_train_batch_end(self, outputs: torch.Tensor, batch: torch.Tensor, batch_idx: int) -> None:
        self.ma_callback.on_train_batch_end(self.trainer, self, outputs, batch, batch_idx)

    # is it really necessary?
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["audio_ema"] = self.audio_ema.state_dict()
        checkpoint["text_ema"] = self.text_ema.state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.audio_ema.load_state_dict(checkpoint["audio_ema"])
        self.text_ema.load_state_dict(checkpoint["text_ema"])

