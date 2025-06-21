import math
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.

    Code stolen from https://github.com/facebookresearch/FFCV-SSL/blob/main/examples/train_ssl.py
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


class CLAPLoss(nn.Module):
    r"""Multimodal contrastive loss as defined in CLIP. This implementation is slightly faster than the ones
    proposed by facebookresearch and lucidrains.
    See https://colab.research.google.com/drive/140rPg9YYzgTtLEQMTO1qtFaPsv5kMgR7?usp=sharing#scrollTo=fqzNULcucAjV
    for a basic benchmark

    Args:
        temperature (float): temperature parameter. If `trainable == True`, it corresponds to the initial temperature.
            If `schedule` is not `None` it corresponds to the maximal temperature.
        trainable (bool): make temperature a trainable parameter
        schedule (str):
    """

    def __init__(
            self,
            temperature: float = 0.1,
            temperature_text: float | None = None,
            temperature_audio: float | None = None,
            trainable: bool = False,
            do_ssl: bool = False,
            do_ssl_weight: float = 0.2
    ):
        super(CLAPLoss, self).__init__()
        # logit_scale_* = - log(temp_*)  <=> 1 / temp_* = exp(logit_scale_*)
        self.logit_scale =  nn.Parameter(
            torch.tensor(temperature_audio or temperature),
            requires_grad=trainable)
        # )
        # self.logit_scale_t = 1/ nn.Parameter(
        #     torch.tensor(temperature_text or temperature),
        #     requires_grad=trainable
        # )
        self._trainable_temperature = trainable
        
        self.do_ssl = do_ssl
        self.do_ssl_weight = do_ssl_weight

        # internal variables
        self._world_size = None

    def forward(self, z_a: torch.Tensor, z_t: torch.Tensor):
        r"""Computation of the contrastive loss

        Args:
            z_a, z_t (torch.Tensor): pairs of embeddings, s.t. (z_a[i], z_t[i]) is a positive pair.
                z_a and z_t must have the same shape (batch_size, embed_dim)
        """
        # Here is just a simple test to check that temperature is updated if it is supposed to be trainable.
        # Indeed it is a common mistake to forget to add the loss parameters to the optimizer.
        z_a = F.normalize(z_a)
        z_t = F.normalize(z_t)
        

        # if self.world_size > 1:
        #     z_a = torch.cat(GatherLayer.apply(z_a), dim=0)
        #     z_t = torch.cat(GatherLayer.apply(z_t), dim=0)



        sim = torch.mm(z_a, z_t.t())  # cosine similarity   
        sim = sim * torch.div(1, self.logit_scale)
        
        
        labels = torch.arange(sim.shape[0],device=sim.device)
        loss_a = F.cross_entropy(sim, labels)
        loss_t = F.cross_entropy(sim.T, labels)
        total_loss = (loss_a + loss_t) / 2
        
        
        if self.do_ssl:
            target_t_sims = torch.mm(z_t, z_t.t())
            target_a_sims = torch.mm(z_a, z_a.t())
            target_t_sims = target_t_sims
            target_a_sims = target_a_sims
            ssl_loss_a = F.binary_cross_entropy_with_logits(target_a_sims, target_t_sims)
            ssl_loss_t = F.binary_cross_entropy_with_logits(target_t_sims, target_a_sims)
            total_loss += self.do_ssl_weight * (ssl_loss_a + ssl_loss_t) / 2
        
        return_ = {
            "loss_a": loss_a,
            "loss_t": loss_t,
            "total_loss": total_loss
        } if not self.do_ssl else {
            "loss_a": loss_a,
            "loss_t": loss_t,
            "ssl_loss_a": ssl_loss_a,
            "ssl_loss_t": ssl_loss_t,
            "total_loss": total_loss
        }


        
        
        return return_
        

    @property
    def world_size(self) -> int:
        if self._world_size is None:
            # compute world size
            self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        return self._world_size
