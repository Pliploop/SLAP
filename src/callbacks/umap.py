import logging

import plotly.express as px
from umap import UMAP

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
import torch
import torch.nn.functional as F

from src.utils.wandb import wandb_only


log = logging.getLogger(__name__)


class UMAPVisualization(pl.Callback):
    def __init__(self, every_n_epochs: int = 1, n_samples: int = 1000):
        self.every_n_epochs = every_n_epochs
        self.logger = None
        self.umap = UMAP(n_components=2)
        self.n_samples = n_samples

    @rank_zero_only
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        for logger in pl_module.loggers:
            if isinstance(logger, WandbLogger):
                self.logger = logger
                break
        if self.logger is None:
            log.warning(f"Tried to use `{self.__class__.__name__}` whereas no WandbLogger has been found. "
                        f"Some of the logging features won't work properly. "
                        f"Loggers: {pl_module.loggers}.")

    def on_validation_epoch_end(self,
                                trainer: pl.Trainer,
                                pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs != 0:
            return
        
        pl_module.plotting_dict = self._gather_plotting_dict(pl_module.plotting_dict)
        
        try:
            for i, plotting_dict in enumerate(pl_module.plotting_dict):
                for plot_ in plotting_dict['plots']:
                    print({
                        k:torch.cat(v).shape for k, v in plot_.items()
                    })  
        except Exception as e:
            print(f'Error: {e}')

        for i, plotting_dict in enumerate(pl_module.plotting_dict):
            self.plot_umap(plotting_dict, dataset_name=str(i))

        pl_module.reset_plotting_dict()
        
    def _gather_plotting_dict(self, plotting_dict):
        # Synchronize across GPUs
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if world_size == 1:
            return plotting_dict
        
        gathered_dicts = []
        for plotting_dict_ in plotting_dict:
            gathered_dict = {
                'key': [],
                'plots': []
            }
            gathered_dict['key'] = self._gather_objects(plotting_dict_['key'])
            for plot_ in plotting_dict_['plots']:
                # plot is a dictionary with keys being names and values being lists of tensors
                gathered_plot = {k: [] for k in plot_.keys()}
                for k, v in plot_.items():
                    gathered_plot[k] = [self._gather_tensors(tensor) for tensor in v]
                gathered_dict['plots'].append(gathered_plot)
            gathered_dicts.append(gathered_dict)
        return gathered_dicts
    
    def _gather_tensors(self, tensor: torch.Tensor) -> torch.Tensor:
        # Gather tensors across all GPUs
        if torch.distributed.is_initialized():
            gathered_tensors = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered_tensors, tensor)
            return torch.cat(gathered_tensors)
        return tensor
    
    def _gather_objects(self, objects: list) -> list:
        # Gather tensors across all GPUs
        if torch.distributed.is_initialized():
            gathered_objects = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(gathered_objects, objects)
            gathered__ = []
            for gathered_ in gathered_objects:
                gathered__+= gathered_
        return gathered__

    @wandb_only
    @rank_zero_only
    def plot_umap(self, plotting_dict, dataset_name: str):
        plotting_key = plotting_dict['key']
        plots = plotting_dict['plots'] # list of dicts with keys being names and values being tensors
        if plotting_key == []:
            return

        fig_dict = {}
        empty_dict = {
            'key': [],
            'plots': []
        }
        
        for plot_ in plots:
            embeds_ = []
            color = []
            text = []
            for c_, e_ in plot_.items():
                embeds_+= e_
                color+= [c_]*len(plotting_key)
                text+= plotting_key
                
            empty_dict['plots'].append({c_: [] for c_ in plot_.keys()})
            embeddings = torch.cat(embeds_)
            embeddings = F.normalize(embeddings, p=2, dim=-1).detach().cpu().numpy() 

            # random sampling
            if embeddings.shape[0] > self.n_samples:
                idx = torch.randperm(embeddings.shape[0])[:self.n_samples]
                embeddings = embeddings[idx]
                color = [color[i] for i in idx]
                text = [text[i] for i in idx]
             

            umap_2d = UMAP(n_components=2)
            umap_2d.fit(embeddings)

            projections = umap_2d.transform(embeddings)

            name = '+'.join(list(plot_.keys()))

            if len(text) > len(projections):
                descriptions_per_embedding, r = divmod(len(text), len(projections))
                if r != 0:
                    log.error(f"Got {len(projections)} embeddings but {len(text)} descriptions for dataset {dataset_name}. "
                              f"This is weird and should be investigated...\n{str(text[:10])}\nSkipping UMAP...")
                    continue

                text = text[::descriptions_per_embedding]
                color = color[::descriptions_per_embedding]
            
            fig_dict[f"UMAP_{dataset_name}/{name}"] = px.scatter(projections, x=0, y=1, hover_name=text, opacity=0.5, color=color)

        self.logger.experiment.log(fig_dict)
    
