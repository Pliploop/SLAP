import math
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F



class BYOLLoss(nn.Module):
    r"""
    """

    def __init__(
            self,
            out_key = "all_loss",
            unimodal: bool = True,
            ssl_weight: float = 0.5
    ):
        super(BYOLLoss, self).__init__()
        self.out_key = out_key
        self.unimodal = unimodal
        self.ssl_weight = ssl_weight # if 1, only multimodal loss, if 0, only unimodal loss
        

    def forward(self, q_a: torch.Tensor, q_t: torch.Tensor, z_a_ema: torch.Tensor, z_t_ema: torch.Tensor):
        r"""Computation of the byol loss
        """
        
        
        q_a = F.normalize(q_a)
        q_t = F.normalize(q_t)
        z_a_ema = F.normalize(z_a_ema)
        z_t_ema = F.normalize(z_t_ema)
        
        a_t_loss = self.forward_single(q_a, z_t_ema)
        t_a_loss = self.forward_single(q_t, z_a_ema)

        if self.unimodal:
            a_a_loss = self.forward_single(q_a, z_a_ema)
            t_t_loss = self.forward_single(q_t, z_t_ema)
        else:
            a_a_loss = torch.zeros_like(a_t_loss)
            t_t_loss = torch.zeros_like(t_a_loss)

        all_loss = self.ssl_weight * (a_t_loss + t_a_loss) / 2 + (1 - self.ssl_weight) * (a_a_loss + t_t_loss) / 2
        
        out_ = {
            "a_t_loss": a_t_loss,
            "t_a_loss": t_a_loss,
            "a_a_loss": a_a_loss,
            "t_t_loss": t_t_loss,
            "all_loss": all_loss,
            "multimodal_loss": (a_t_loss + t_a_loss) / 2,
            "unimodal_loss": (a_a_loss + t_t_loss) / 2
        }
        
        out_["total_loss"] = out_[self.out_key]
        return out_
        
    def forward_single(self,q, z):
        loss = 2 - 2 * (q * z).sum(dim=-1)
        return loss.mean()