import warnings
from typing import Tuple

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
import torch

# Linear logistic classifier from sklearn
from sklearn.linear_model import LogisticRegression
import numpy as np
from typing import List


class ModalityGap(pl.Callback):
    def __init__(self, linsep_split = [0.8,0.2], every_n_epochs: int = 1):
        self.linsep_split = linsep_split
        self.every_n_epochs = every_n_epochs
        self.logger = None


    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for logger in pl_module.loggers:
            if isinstance(logger, WandbLogger):
                self.logger = logger
                break
        if self.logger is None:
            warnings.warn(f"Tried to use `{self.__class__.__name__}` whereas no WandbLogger has been found. "
                          f"Some of the logging features won't work properly. "
                          f"Loggers: {pl_module.loggers}.")

    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.gather_and_display_metrics(pl_module, "val", device = next(pl_module.parameters()).device)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.gather_and_display_metrics(pl_module, "test", device = next(pl_module.parameters()).device)
        
    def gather_and_display_metrics(self, pl_module: pl.LightningModule, step_name: str, device: torch.device) -> None:
        # Gather `retrieval_dict` across all GPUs
        pl_module.modality_gap = self._gather_gap_dict(pl_module.modality_gap, device)

        for key_ in pl_module.modality_gap[0].keys():
            self.display_metrics(pl_module, key_, step_name)
        pl_module.reset_gap_dict()

    def _gather_gap_dict(self, gap_dict: List[dict], device: torch.device) -> List[dict]:
        # Synchronize across GPUs
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if world_size == 1:
            return gap_dict  # No need to gather if single GPU

        gathered_dicts = []
        for data in gap_dict:
            gathered_data = {k: {"audio": [], "text": []} for k in data.keys()}

            for key, tensors in data.items():
                key_gathered = [self._gather_tensors(tensor) for tensor in tensors['audio']]
                query_gathered = [self._gather_tensors(tensor) for tensor in tensors['text']]            
                gathered_data[key]['audio'] = key_gathered
                gathered_data[key]['text'] = query_gathered

            gathered_dicts.append(gathered_data)
        return gathered_dicts

    def _gather_tensors(self, tensor: torch.Tensor) -> torch.Tensor:
        # Gather tensors across all GPUs
        if torch.distributed.is_initialized():
            gathered_tensors = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered_tensors, tensor)
            return torch.cat(gathered_tensors)
        return tensor

    def display_metrics(self, pl_module: pl.LightningModule, key_ : str, step_name: str) -> None:
        for i, modality_gap in enumerate(pl_module.modality_gap):
            if len(modality_gap[key_]['audio']) == 0:
                continue

            audio = torch.cat(modality_gap[key_]['audio']).cpu() # to not explode the GPU
            text = torch.cat(modality_gap[key_]['text']).cpu()
            
            sn = step_name + '_' + str(i)
            
            centroid_distance = self.compute_distance_centroids(audio, text)
            linear_separability = self.compute_linsep(audio, text)
            
            metrics = {
                f"Modality gap/Centroid Distance/{sn}/{key_}": centroid_distance,
                f"Modality gap/Linear Separability/{sn}/{key_}": linear_separability
            }


            device = next(pl_module.parameters()).device
            for k, v in metrics.items():
                if hasattr(v, "to"):
                    metrics[k] = v.to(device)

            pl_module.log_dict(metrics, sync_dist=True)


    @staticmethod
    def compute_distance_centroids(audio_embeddings, text_embeddings):
        audio_centroid = audio_embeddings.mean(dim=0)
        text_centroid = text_embeddings.mean(dim=0)
        return torch.norm(audio_centroid - text_centroid, p=2)
    
    def compute_linsep(self, audio_embeddings, text_embeddings):
        audio_embeddings, text_embeddings = audio_embeddings.cpu().numpy(), text_embeddings.cpu().numpy()
        embeddings = np.concatenate((audio_embeddings, text_embeddings), axis=0)
        labels = np.concatenate((np.zeros(audio_embeddings.shape[0]), np.ones(text_embeddings.shape[0])), axis=0)
        # random split using linsep_split
        indices = np.random.permutation(embeddings.shape[0])
        split = int(self.linsep_split[0] * embeddings.shape[0])
        train_indices, test_indices = indices[:split], indices[split:]
        train_embeddings, train_labels = embeddings[train_indices], labels[train_indices]
        test_embeddings, test_labels = embeddings[test_indices], labels[test_indices]
        # train logistic regression
        clf = LogisticRegression(random_state=0).fit(train_embeddings, train_labels)
        # test logistic regression
        return clf.score(test_embeddings, test_labels)
        # TODO : should this really be random split? or should we use the same split for all epochs?
        
