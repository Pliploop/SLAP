r"""Custom functions to register by Hydra"""
from typing import Callable, Dict

from dora import get_xp
from omegaconf import OmegaConf


def register_custom_resolvers(extra_resolvers: Dict[str, Callable] = None):
    """Wrap your main function with this.
    You can pass extra kwargs, e.g. `version_base` introduced in 1.2.
    """
    extra_resolvers = extra_resolvers or {}
    for name, resolver in extra_resolvers.items():
        OmegaConf.register_new_resolver(name, resolver)


def dora_resolver(var: str) -> str:
    obj, attr = var.split('.')
    if obj == "xp":
        try:
            xp = get_xp()
        except RuntimeError as e:
            if str(e) == "Not in a xp!":
                return ""
            raise e

        return str(getattr(xp, attr))

    raise ValueError("Unrecognized variable '{}'".format(var))


def effective_lr(base_lr: float, batch_size: int) -> float:
    return base_lr * batch_size / 256


def register_resolvers():
    register_custom_resolvers({
        "dora": dora_resolver
    })
