import warnings
from typing import Tuple, List

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
import torch

from src.utils.wandb import wandb_only

class RetrievalEvaluation(pl.Callback):
    def __init__(self, thresholds=(1, 5, 10), every_n_epochs: int = 1, distance='cosine'):
        self.thresholds = thresholds
        self.every_n_epochs = every_n_epochs
        self.distance = distance
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
        pl_module.retrieval_dict = self._gather_retrieval_dict(pl_module.retrieval_dict, device)

        for key_ in pl_module.retrieval_dict[0].keys():
            self.display_metrics(pl_module, key_, step_name)
        pl_module.reset_retrieval_dict()

    def _gather_retrieval_dict(self, retrieval_dict: List[dict], device: torch.device) -> List[dict]:
        # Synchronize across GPUs
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if world_size == 1:
            return retrieval_dict  # No need to gather if single GPU

        gathered_dicts = []
        for data in retrieval_dict:
            gathered_data = {k: {"key": [], "query": []} for k in data.keys()}

            for key, tensors in data.items():
                key_gathered = [self._gather_tensors(tensor) for tensor in tensors['key']]
                query_gathered = [self._gather_tensors(tensor) for tensor in tensors['query']]            
                gathered_data[key]['key'] = key_gathered
                gathered_data[key]['query'] = query_gathered

            gathered_dicts.append(gathered_data)
        return gathered_dicts

    def _gather_tensors(self, tensor: torch.Tensor) -> torch.Tensor:
        # Gather tensors across all GPUs
        if torch.distributed.is_initialized():
            gathered_tensors = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered_tensors, tensor)
            return torch.cat(gathered_tensors)
        return tensor

    
    def display_metrics(self, pl_module: pl.LightningModule, key_: str, step_name: str) -> None:
        for i, retrieval_dict in enumerate(pl_module.retrieval_dict):
            if len(retrieval_dict[key_]['key']) == 0:
                continue

            key = torch.cat(retrieval_dict[key_]['key']).cpu()
            query = torch.cat(retrieval_dict[key_]['query']).cpu()

            similarities = torch.mm(query, key.t()) if self.distance == 'cosine' else None
            similarities = -torch.cdist(query, key, p=2) if self.distance == 'euclidean' else similarities

            num_embeddings_1, num_embeddings_2 = similarities.size()
            recalls, ranks = self.compute_recall(similarities.t())
            auc = self.compute_auc(recalls)
            normalized_ranks = ranks.float().div_(ranks.max())

            sn = step_name + '_' + str(i)
            metrics = {f"Recall/{sn}/{key_} R@{k:d}": recalls[k] for k in self.thresholds}
            metrics[f"Metrics/{sn}/{key_} AUC"] = auc
            metrics[f"Metrics/{sn}/{key_} mean Rank"] = normalized_ranks.mean()
            metrics[f"Metrics/{sn}/{key_} median Rank"] = normalized_ranks.median()

            #send everything to the pl_module device
            
            device = next(pl_module.parameters()).device
            for k, v in metrics.items():
                if hasattr(v, "to"):
                    metrics[k] = v.to(device) #TODO this is kind of ugly

            pl_module.log_dict(metrics, sync_dist=True)

            self._plot_wandb_line(f"Retrieval/{sn}/{key_} R@K", {
                f"K ({sn})": torch.arange(51),
                f"{key_} R@K": recalls[:51]
            })

    @staticmethod
    def compute_recall(similarities: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        num_src_embeddings, num_tgt_embeddings = similarities.size()
        device = similarities.device
        

        true_indices = torch.arange(num_src_embeddings, device=device).unsqueeze(1)
        sorted_indices = similarities.argsort(descending=True)

        if num_src_embeddings < num_tgt_embeddings:
            tgt_per_src, r = divmod(num_tgt_embeddings, num_src_embeddings)
            assert r == 0
            sorted_indices = sorted_indices.div_(tgt_per_src, rounding_mode="floor")

        else:
            src_per_tgt, r = divmod(num_src_embeddings, num_tgt_embeddings)
            assert r == 0
            true_indices.div_(src_per_tgt, rounding_mode="floor")

        ranks = (sorted_indices == true_indices).long().argmax(dim=1)  # argmax?

        recalls = torch.zeros(num_tgt_embeddings + 1, dtype=torch.long, device=device)
        values, counts = torch.unique(ranks, return_counts=True)
        recalls[values + 1] = counts
        return recalls.cumsum(dim=0).float().div_(num_src_embeddings), ranks

    @staticmethod
    def compute_auc(values: torch.Tensor) -> torch.Tensor:
        return torch.mean(values[:-1] + values[1:]) / 2

    @wandb_only
    @rank_zero_only
    def _plot_wandb_line(self, title, elems):
        import wandb
        assert len(elems) == 2, f"You should provide x and y, got {elems}"
        data = [tensor.cpu().numpy() for tensor in elems.values()]
        cols = list(elems.keys())
        table = wandb.Table(data=list(zip(*data)), columns=cols)
        self.logger.experiment.log({title: wandb.plot.line(table, *cols)})

    def _plot_wandb_cdf(self, title, metric, sorted_elems):
        self._plot_wandb_line(title, {
            metric: sorted_elems,
            "Cumulative Density Function": torch.arange(1, len(sorted_elems) + 1) / len(sorted_elems)
        })

    @wandb_only
    def _plot_wandb_histogram(self, title: str, values: torch.Tensor):
        import wandb
        self.logger.experiment.log({title: wandb.Histogram(values.cpu())})