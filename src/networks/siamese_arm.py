import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


log = logging.getLogger(__name__)


class SiameseArm(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            projector: nn.Module | None = None,
            transform: nn.Module | None = None,
            normalize_representations: bool = False,
            normalize_projections: bool = False,
            freeze_encoder: bool = False,
            checkpoint_path: str | None = None
    ):
        super(SiameseArm, self).__init__()

        # networks
        self.encoder = encoder
        self.projector = projector
        self.transform = transform or nn.Identity()

        # normalizations
        self.normalize_y = F.normalize if normalize_representations else nn.Identity()
        self.normalize_z = F.normalize if normalize_projections else nn.Identity()

        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
                try:
                    self.load_state_dict(checkpoint)
                except RuntimeError as e:
                    log.warning(e)
                    self.load_state_dict(checkpoint, strict=False)
            else:
                log.warning(f"Checkpoint not found at {checkpoint_path}")

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for p, param in self.encoder.named_parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.freeze_encoder:
            self.encoder.eval()  # Lightning by default puts everything in training mode during training

        y = self.encoder(x)
        
        if self.projector is None:
            return y, None, None

        z = self.projector(y)
        q = self.transform(z)
        
        
        y = self.normalize_y(y)
        z = self.normalize_z(z)
        q = self.normalize_z(q)

        return y, z, q

    @property
    def out_features(self):
        return self.projector.weight.size(1)
