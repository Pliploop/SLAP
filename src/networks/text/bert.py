import logging

import torch
import torch.nn as nn

from transformers import AutoModel, AutoConfig


log = logging.getLogger(__name__)


class HuggingFaceTextEncoder(nn.Module):
    def __init__(self, model_name: str, checkpoint_path: str | None = None, load_pretrained: bool = False):
        super().__init__()
        self.output_key = "pooler_output"
        self.needs_pool = False

        if model_name.startswith("bert-"):
            from transformers import BertTokenizer, BertModel
            tokenizer_cls = BertTokenizer
            model_cls = BertModel

        elif model_name.startswith("roberta-"):
            from transformers import RobertaTokenizer, RobertaModel
            tokenizer_cls = RobertaTokenizer
            model_cls = RobertaModel
        
        elif model_name.startswith("answerdotai/ModernBERT"):
            from transformers import AutoTokenizer, ModernBertModel
            tokenizer_cls = AutoTokenizer
            model_cls = ModernBertModel
            self.output_key = "last_hidden_state"
            self.needs_pool = True

        else:
            raise ValueError(f"Model name `{model_name}` not supported")

        self.tokenizer = tokenizer_cls.from_pretrained(model_name)
        self.model = model_cls.from_pretrained(model_name) if load_pretrained else AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        log.info(f'Loaded model {model_name} pretrained: {load_pretrained}')

        if checkpoint_path and load_pretrained:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
            try:
                self.load_state_dict({k: v for k, v in checkpoint.items()})
            except RuntimeError as e:
                log.warning(e)
                self.load_state_dict({k: v for k, v in checkpoint.items()}, strict=False)
        
        self.register_buffer("_device", torch.zeros(1), persistent=False)

    def forward(self, tokens):
        
        if isinstance(tokens, str) or isinstance(tokens, list):
            tokens = self.tokenizer(tokens, return_tensors="pt", padding=True, truncation=True)
            tokens = tokens.to(self._device.device)
        
        out = self.model(**tokens)[self.output_key]

        return out.mean(dim=-2) if self.needs_pool else out
