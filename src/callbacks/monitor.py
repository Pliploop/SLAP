from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT


class TemperatureMonitor(pl.Callback):
    def on_train_batch_end(self,
                           trainer: pl.Trainer,
                           pl_module: pl.LightningModule,
                           outputs: STEP_OUTPUT,
                           batch: Any,
                           batch_idx: int) -> None:
        pl_module.log_dict({"params/" + name: param.cpu().item() for name, param in pl_module.loss_fn.named_parameters()})