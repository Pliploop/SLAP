from typing import Type
import torch
import torch.nn as nn

from .base import BaseModule, BaseEMAModule


class SLAP(BaseEMAModule):

    def __init__(
        self,
        audio_encoder: nn.Module,
        text_encoder: nn.Module,
        loss_fn: nn.Module,
        optimizer: Type,
        scheduler: Type | None = None,
        ma_callback=None,
        compile: bool = False,
        **kwargs,
    ):
        super().__init__(
            audio_encoder,
            text_encoder,
            loss_fn,
            optimizer,
            scheduler,
            ma_callback=ma_callback,
            compile=compile,
            **kwargs,
        )

        self.plotting_dict = [{
            "plots": [
                {"Audio embeddings": []},
                {"Text embeddings": []},
                {
                    "Audio projections": [],
                    "Text projections": [],
                    "A->T predictions": [],
                    "T->A predictions": [],
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
            "A->T Predictions": {
                "key": [],
                "query": [],
            },
            "T->A Predictions": {
                "key": [],
                "query": [],
            },
        } for _ in range(4)]  # TODO: not being a clebard
        
        self.modality_gap = [{
            "predictions": {
                "audio": [],
                "text": [],
            },
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

        with torch.no_grad():
            ya_ema, za_ema, qa_ema = self.audio_ema(xa)  # q is after predictor
            yt_ema, zt_ema, qt_ema = self.text_ema(xt)

        # compute loss
        loss_dict = self.loss_fn(qt, qa, za_ema, zt_ema)
        # self.save_embeddings_plot(ya, za, yt, zt, qa, qt, descriptions)
        # self.save_embeddings_metric(ya, za, yt, zt, qa, qt, descriptions)

        # log metrics
        self.log_dict({f"loss/train/{k}": v for k, v in loss_dict.items()})

        return loss_dict["total_loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
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

        with torch.no_grad():
            ya_ema, za_ema, qa_ema = self.audio_ema(xa)
            yt_ema, zt_ema, qt_ema = self.text_ema(xt)


        loss_dict = self.loss_fn(qt, qa, za_ema, zt_ema)
        self.log_dict({f"loss/val/{k}": v for k, v in loss_dict.items()})

        self.save_embeddings_metric(dataloader_idx, ya, za, yt, zt, qa, qt, descriptions)
        self.save_embeddings_plot(dataloader_idx, ya, za, yt, zt, qa, qt, descriptions)
        self.save_embeddings_gap(dataloader_idx, ya, za, yt, zt, qa, qt, descriptions)

        return za, zt, qa, qt, descriptions
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
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

        self.save_embeddings_metric(dataloader_idx, ya, za, yt, zt, qa, qt, descriptions)
        self.save_embeddings_plot(dataloader_idx, ya, za, yt, zt, qa, qt, descriptions)
        self.save_embeddings_gap(dataloader_idx, ya, za, yt, zt, qa, qt, descriptions)

        return za, zt, qa, qt, descriptions

    # TODO: looks quite redundant, maybe one can simplify this?
    def save_embeddings_plot(self, idx, ya, za, yt, zt, qa, qt, descriptions):
        r"""Save all embeddings in attributes."""
        self.plotting_dict[idx]["plots"][0]["Audio embeddings"].append(ya.float())
        self.plotting_dict[idx]["plots"][1]["Text embeddings"].append(yt.float())
        self.plotting_dict[idx]["plots"][2]["Audio projections"].append(za.float())
        self.plotting_dict[idx]["plots"][2]["Text projections"].append(zt.float())
        self.plotting_dict[idx]["plots"][2]["T->A predictions"].append(qt.float())
        self.plotting_dict[idx]["plots"][2]["A->T predictions"].append(qa.float())
        self.plotting_dict[idx]["key"] += [d for d in descriptions]

    def save_embeddings_metric(self, idx, ya, za, yt, zt, qa, qt, descriptions):
        r"""Save all embeddings in attributes."""

        self.retrieval_dict[idx]["A->T"]["key"].append(zt.float())
        self.retrieval_dict[idx]["A->T"]["query"].append(za.float())
        self.retrieval_dict[idx]["T->A"]["key"].append(za.float())
        self.retrieval_dict[idx]["T->A"]["query"].append(zt.float())
        self.retrieval_dict[idx]["A->T Predictions"]["key"].append(qt.float())
        self.retrieval_dict[idx]["A->T Predictions"]["query"].append(qa.float())
        self.retrieval_dict[idx]["T->A Predictions"]["key"].append(qa.float())
        self.retrieval_dict[idx]["T->A Predictions"]["query"].append(qt.float())


    def save_embeddings_gap(self, idx, ya, za, yt, zt, qa, qt, descriptions):
        r"""Save all embeddings in attributes."""
        self.modality_gap[idx]["predictions"]["audio"].append(qa.float())
        self.modality_gap[idx]["predictions"]["text"].append(qt.float())
        self.modality_gap[idx]["projections"]["audio"].append(za.float())
        self.modality_gap[idx]["projections"]["text"].append(zt.float())

    def reset_retrieval_dict(self):
        self.retrieval_dict = [{
            "A->T": {"key": [], "query": []},
            "T->A": {
                "key": [],
                "query": [],
            },
            "A->T Predictions": {
                "key": [],
                "query": [],
            },
            "T->A Predictions": {
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
                    "A->T predictions": [],
                    "T->A predictions": [],
                },
            ],
            "key": [],
        } for _ in range(4)]
        
    def reset_gap_dict(self):
        self.modality_gap = [{
            "predictions": {
                "audio": [],
                "text": [],
            },
            'projections': {
                'audio': [],
                'text': [],
            },
            } for _ in range(4)]

