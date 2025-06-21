from typing import Type
import torch
import torch.nn as nn

from .base import BaseModule


class CLAP(BaseModule):
    
    
    def __init__(self,
                 audio_encoder: nn.Module,
                 text_encoder: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: Type,
                 scheduler: Type | None = None,
                 compile: bool = False,
                 **kwargs):
        super().__init__(audio_encoder, text_encoder, loss_fn, optimizer, scheduler, compile, **kwargs)
        
        
        self.plotting_dict = [{
            "plots": [
                {"Audio embeddings": []},
                {"Text embeddings": []},
                {
                    "Audio projections": [],
                    "Text projections": [],
                },
            ],
            "key": [],
        } for _ in range(4)]

        self.retrieval_dict = [{
            "A->T": {"key": [], "query": []},
            "T->A": {
                "key": [],
                "query": [],
            },
        } for _ in range(4)]  # TODO: not being a clebard
        
        
        self.modality_gap = [{
            'projections': {
                'audio': [],
                'text': [],
            },
            } for _ in range(4)]
        
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        xa, xt, descriptions = batch

        # encode text and audio
        ya, za, qa = self.audio_encoder(xa)
        yt, zt, qt = self.text_encoder(xt)

        # compute loss
        loss_dict = self.loss_fn(za, zt)

        # log metrics
        self.log_dict({f"loss/train/{k}": v for k, v in loss_dict.items()})

        return loss_dict["total_loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        r"""Here we only compute embeddings and projections for all pairs of inputs.
        For any additional computation (metrics, visualization...) we use callbacks.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): input pair of audio-text
            batch_idx (int): index of the batch (unused)

        Returns:
            Audio embedding
            Audio projection
            Text embedding
            Text projection
        """
        xa, xt, descriptions = batch
        ya, za, qa = self.audio_encoder(xa)
        yt, zt, qt = self.text_encoder(xt)
        
        self.save_embeddings_metric(dataloader_idx, ya, za, yt, zt, descriptions)
        self.save_embeddings_plot(dataloader_idx, ya, za, yt, zt, descriptions)
        self.save_embeddings_gap(dataloader_idx, ya, za, yt, zt, descriptions)

        loss_dict = self.loss_fn(za, zt)
        self.log_dict({f"loss/val/{k}": v for k, v in loss_dict.items()})
        
        return za, zt, descriptions
    
    
    def test_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Tensor:
        r"""Here we only compute embeddings and projections for all pairs of inputs.
        For any additional computation (metrics, visualization...) we use callbacks.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): input pair of audio-text
            batch_idx (int): index of the batch (unused)

        Returns:
            Audio embedding
            Audio projection
            Text embedding
            Text projection
        """
        xa, xt, descriptions = batch
        ya, za, qa = self.audio_encoder(xa)
        yt, zt, qt = self.text_encoder(xt)
        
        # loss_dict = self.loss_fn(za, zt)
        # self.log_dict({f"val_loss/{k}": v for k, v in loss_dict.items()})
        
        
        self.save_embeddings_metric(dataloader_idx, ya, za, yt, zt, descriptions)
        self.save_embeddings_plot(dataloader_idx, ya, za, yt, zt, descriptions)
        self.save_embeddings_gap(dataloader_idx, ya, za, yt, zt, descriptions)
        
        return za, zt, descriptions
    
    def save_embeddings_plot(self, idx, ya, za, yt, zt, descriptions):
        r"""Save all embeddings in attributes."""
        self.plotting_dict[idx]["plots"][0]["Audio embeddings"].append(ya.float())
        self.plotting_dict[idx]["plots"][1]["Text embeddings"].append(yt.float())
        self.plotting_dict[idx]["plots"][2]["Audio projections"].append(za.float())
        self.plotting_dict[idx]["plots"][2]["Text projections"].append(zt.float())
        self.plotting_dict[idx]["key"] += [d for d in descriptions]

    def save_embeddings_metric(self, idx, ya, za, yt, zt, descriptions):
        r"""Save all embeddings in attributes."""

        self.retrieval_dict[idx]["A->T"]["key"].append(zt.float())
        self.retrieval_dict[idx]["A->T"]["query"].append(za.float())
        self.retrieval_dict[idx]["T->A"]["key"].append(za.float())
        self.retrieval_dict[idx]["T->A"]["query"].append(zt.float())
    
    def save_embeddings_gap(self, idx, ya, za, yt, zt, descriptions):
        r"""Save all embeddings in attributes."""
        self.modality_gap[idx]['projections']['audio'].append(za.float())
        self.modality_gap[idx]['projections']['text'].append(zt.float())
        
    def reset_gap_dict(self):
        self.modality_gap = [{
            'projections': {
                'audio': [],
                'text': [],
            },
            } for _ in range(4)]

    def reset_retrieval_dict(self):
        self.retrieval_dict = [{
            "A->T": {"key": [], "query": []},
            "T->A": {
                "key": [],
                "query": [],
            },
        } for _ in range(4)]

    def reset_plotting_dict(self):
        self.plotting_dict = [{
            "plots": [
                {"Audio embeddings": []},
                {"Text embeddings": []},
                {
                    "Audio projections": [],
                    "Text projections": [],
                },
            ],
            "key": [],
        } for _ in range(4)]