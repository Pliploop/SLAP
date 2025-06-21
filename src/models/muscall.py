from typing import Type
import torch
import torch.nn as nn

from .base import BaseModule

import abc
import logging
from copy import deepcopy
from typing import Tuple, Type, Dict, Any

import torch
import torch.nn as nn

import lightning.pytorch as pl


from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.clip.tokenization_clip import CLIPTokenizer
import torch.nn.functional as F
from math import ceil

from pathlib import Path
import yaml

from easydict import EasyDict

from transformers import CLIPTextModel
 

from src.models.muscall.modules.textual_heads import TextTransformer
from src.models.muscall.modules.audio_backbones import ModifiedResNet



def load_yaml_config(path_to_config):
    """Loads yaml configuration settings as an EasyDict object."""
    path_to_config = Path(path_to_config)
    assert path_to_config.is_file(), f'{path_to_config} not found, cwd={Path(".").resolve()}'
    with open(path_to_config) as f:
        yaml_contents = yaml.safe_load(f)
    cfg = EasyDict(yaml_contents)
    return cfg


class MusCALL(pl.LightningModule, metaclass=abc.ABCMeta):

    def __init__(self,
                 weight_file: str,
                 config_file: str,
                 tokenizer: str,
                 loss_fn: nn.Module,
                 optimizer: Type,
                 audio_encoder = None,
                 text_encoder = None,
                 embed_dim = None,
                 scheduler: Type | None = None,
                 compile: bool = False,
                 
                 **kwargs):

        super(MusCALL, self).__init__()
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


        config = load_yaml_config(config_file).model_config

        audio_config = config.audio
        text_config = config.text

        projection_dim = config.projection_dim
        audio_dim = audio_config.hidden_size
        text_dim = text_config.hidden_size

        if config.audio.model == "ModifiedResNet":
            self.audio_backbone = ModifiedResNet(audio_config)
        if config.text.model == "TextTransformer":
            self.textual_head = TextTransformer(text_config)
        elif config.text.model == "CLIPTextModel":
            pretrained_model = config.text.pretrained
            self.textual_head = CLIPTextModel.from_pretrained(pretrained_model)

        self.audio_projection = nn.Linear(
            audio_dim, projection_dim, bias=False)
        self.text_projection = nn.Linear(text_dim, projection_dim, bias=False)

        self._build_tokenizer(tokenizer)

        self._init_weights(weight_file)

    def _init_weights(self, weight_file) -> None:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("! Didn't implement checkpoint loading !")
        print("! see evar/evar/ar_muscall.py         !")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # load weights from checkpoint
        
        if weight_file is None:
            return
        ## if s3 is in the weight file, download it to a temp location and delete once the weights are loaded
        s3_flag = "s3://" in str(weight_file)
        
        if s3_flag:
            import boto3
            import tempfile
            import os
            import shutil

            s3 = boto3.client('s3')
            bucket, key = weight_file.split("s3://")[1].split("/", 1)
            print(f"Downloading weights from s3://{bucket}/{key}")
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfile = os.path.join(tmpdir, key)
                os.makedirs(os.path.dirname(tmpfile), exist_ok=True)
                s3.download_file(bucket, key, tmpfile)
                print(f"Downloaded weights to {tmpfile}")
                state_dict = torch.load(tmpfile, map_location=torch.device(
                    'cpu'), weights_only=False)["state_dict"]
        else:
            
            state_dict = torch.load(weight_file, map_location=torch.device(
                'cpu'), weights_only=False)["state_dict"]
        
        self.load_state_dict(state_dict, strict=False)

    def _build_tokenizer(self, tokenizer_name):
        # using tolenizers from pretrained models to reuse their vocab
        if tokenizer_name == "berttokenizer":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif tokenizer_name == "cliptokenizer":
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
        else:
            raise ValueError(
                "{} is not supported. Please provide a valid tokenizer.".format(
                    self.config.text.tokenizer
                )
            )


    def encode_audio(self, audio):
        audio_y = self.audio_backbone(audio, spec_ = True)
        audio_features = self.audio_projection(audio_y)
        return audio_y, audio_features

    def encode_text(self, text):
        input_ids, _, attention_mask = zip(
            *[self.get_text_input(caption) for caption in text])

        input_ids = torch.stack(input_ids).cuda()
        attention_mask = torch.stack(attention_mask).cuda()

        if isinstance(self.textual_head, TextTransformer):
            text_features = self.textual_head(input_ids, attention_mask)
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            pooled_outout = text_features[
                torch.arange(text_features.shape[0]), input_ids.argmax(dim=-1)
            ]
        elif isinstance(self.textual_head, CLIPTextModel):
            outputs = self.textual_head(input_ids, attention_mask)
            pooled_outout = outputs.pooler_output

        text_features = self.text_projection(pooled_outout)
        return pooled_outout,text_features

    def get_input_ids(self, caption):
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Input IDs are obtained by tokenizing the string input, adding special tokens and then converting the sequence to IDs.
        For e.g., if using BertTokenizer, X -->[CLS] X [SEP] --> [101, X_i, 102]

        Same as doing self.convert_tokens_to_ids(self.tokenize(text)).

        """
        input_ids = self.tokenizer.encode(
            caption, max_length=77, truncation=True
        )
        return input_ids

    def get_text_input(self, caption):
        """Build text model input."""
        input_ids = self.get_input_ids(caption)

        input_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        while len(input_ids) < 77:
            input_ids.append(0)
            input_type_ids.append(0)
            attention_mask.append(0)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_type_ids = torch.tensor(input_type_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return input_ids, input_type_ids, attention_mask

    def compute_similarity(self, audio_embeddings, text_embeddings):
        za = F.normalize(audio_embeddings)
        zt = F.normalize(text_embeddings)

        return za @ zt.t()

    def forward(self, batch_audio):
        if hasattr(self, 'lms_mode'):
            x = self.encode_frames_lms(batch_audio)
        else:
            x = self.encode_frames(batch_audio)

        return x.mean(dim=-1)  # [B, D, T] -> [B, D]


    def training_step(self, batch, batch_idx) -> torch.Tensor:
        xa, xt, descriptions = batch

        # encode text and audio
        ya, za = self.encode_audio(xa)
        yt, zt = self.encode_text(descriptions)
        
        
        za,zt = za/za.norm(dim=-1, keepdim=True), zt/zt.norm(dim=-1, keepdim=True)

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
        ya, za = self.encode_audio(xa)
        yt, zt = self.encode_text(descriptions)
        
        za,zt = za/za.norm(dim=-1, keepdim=True), zt/zt.norm(dim=-1, keepdim=True)

        self.save_embeddings_metric(
            dataloader_idx, ya, za, yt, zt, descriptions)
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
        
        
        ya, za = self.encode_audio(xa)
        yt, zt = self.encode_text(descriptions)
        
        za,zt = za/za.norm(dim=-1, keepdim=True), zt/zt.norm(dim=-1, keepdim=True)

        # loss_dict = self.loss_fn(za, zt)
        # self.log_dict({f"val_loss/{k}": v for k, v in loss_dict.items()})

        self.save_embeddings_metric(
            dataloader_idx, ya, za, yt, zt, descriptions)
        self.save_embeddings_plot(dataloader_idx, ya, za, yt, zt, descriptions)
        self.save_embeddings_gap(dataloader_idx, ya, za, yt, zt, descriptions)

        return za, zt, descriptions

    def save_embeddings_plot(self, idx, ya, za, yt, zt, descriptions):
        r"""Save all embeddings in attributes."""
        self.plotting_dict[idx]["plots"][0]["Audio embeddings"].append(
            ya.float())
        self.plotting_dict[idx]["plots"][1]["Text embeddings"].append(
            yt.float())
        self.plotting_dict[idx]["plots"][2]["Audio projections"].append(
            za.float())
        self.plotting_dict[idx]["plots"][2]["Text projections"].append(
            zt.float())
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
