import abc

import torch


class Tokenizer(abc.ABC):
    def __call__(self, inputs):
        return self.tokenize(inputs)

    @abc.abstractmethod
    def tokenize(self, inputs):
        pass


class RandomTokenizer(Tokenizer):
    def tokenize(self, inputs):
        return torch.randn(len(inputs), 512)


class HuggingFaceTokenizer(Tokenizer):
    r"""Wrapper around the utils.tokenizer. Class-based implementation makes it easier to use from YAML config file.
    """
    def __init__(
            self,
            max_sequence_length: int = 512,
            truncation: str = "longest_first",
            padding: str = "longest",
            return_special_tokens_mask: bool = False,
            return_attention_mask: bool = True
    ):
        # the tokenizer is instantiated in model.on_fit_start()
        self.tokenizer = None

        self.tokenizer_options = dict(
            truncation=truncation,
            padding=padding,
            max_length=max_sequence_length,
            return_special_tokens_mask=return_special_tokens_mask,
            return_attention_mask=return_attention_mask
        )

    def tokenize(self, inputs, **kwargs):
        r"""

        Args:
            inputs (List[str]): batch of text to tokenize
        """
        options = {**self.tokenizer_options, **kwargs}
        encoded = self.tokenizer(inputs, **options)
        return dict(
            input_ids=torch.tensor(encoded.data["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(encoded.data["attention_mask"], dtype=torch.long)  # why not dtype bool?
        )
        
    def update_options(self, options):
        self.tokenizer_options = options
