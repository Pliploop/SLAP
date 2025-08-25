import logging
import random
from pathlib import Path
from typing import Any, List, Tuple
from tqdm import trange

import numpy as np
import pandas as pd
from omegaconf import ListConfig, OmegaConf

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from lightning import LightningDataModule

from src.tokenizer.base import Tokenizer
from src.utils.running_stats import OnlineStatsCalculator
import soundfile as sf
import os

import torchaudio.functional as F


log = logging.getLogger(__name__)


def collate_audio_text(data):
    return zip(*data)


class AudioCSVDataset(Dataset):
    def __init__(self,
                 csv_path: str | Path | List[str],
                 num_frames: int,
                 split: str | None = None,
                 root_path: str | None = None,
                 audio_ = True,
                 target_sr: int = 16000,
                 target_frames: int = 320000,
                 multi_caption: str = "random") -> None:
        self.num_frames = num_frames
        
        self.audio = audio_
        self.target_sr = target_sr

        if isinstance(csv_path, list):
            assert root_path is not None, "Please specify the root_path for the dataset"
            self.root_path = Path(root_path)

            df = pd.concat([self._load_dataframe(p) for p in csv_path])

        else:
            if root_path is None:
                self.root_path = Path(csv_path).parents[1]
            else:
                self.root_path = Path(root_path)

            df = self._load_dataframe(csv_path)
        
        if split is not None:
            filter = df["set"] == split

            # when there is no test set we test on the validation set
            if split == "test" and not filter.any():
                filter = df["set"] == "val"
            df = df.loc[filter]

        self.df = df[["npy_path", "caption"]]

        self.multi_caption = multi_caption
        
        log.info(f"Loaded {len(self.df)} tracks from {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, List[str]]:
        r"""

        Args:
            idx (int): Index of the track to load

        Returns:
            torch.Tensor: output spectrogram (or audio if self.audio)
        """
        npy_path, caption = tuple(self.df.iloc[idx])

        # handle multiple captions
        if caption.startswith("!!"):
            captions = caption[2:].split('|')
            if self.multi_caption == "random":
                caption = [random.choice(captions)]
            elif self.multi_caption == "all":
                caption = captions
        else:
            caption = [caption]

        spectrogram = self._load_spectrogram(npy_path) if not self.audio else self._load_audio(npy_path, self.target_sr)

        return spectrogram, caption


    def _load_dataframe(self, csv_path: Path):
        df = pd.read_csv(csv_path, usecols=["npy_path", "caption", "set"])
        return df.drop_duplicates(ignore_index=True)

    def _load_spectrogram(self, npy_path: str):
        npy_path = u'{}'.format(npy_path).replace('"', '')
        memmap = np.load(self.root_path / npy_path, mmap_mode='r')

        spec_size = memmap.shape[-1]
        if self.num_frames is not None and spec_size > self.num_frames:
            start_idx = random.randint(0, spec_size - self.num_frames - 1)
            return torch.from_numpy(memmap[..., start_idx: start_idx + self.num_frames].copy())

        return torch.from_numpy(memmap.copy())

    def _load_audio(self, npy_path: str, target_sr: int, target_frames: int = 320000):
        ## replace npy with or mp3 depending on whether the wav or mp3 file exists
        all_path = self.root_path / npy_path
        all_path = str(all_path)
        if os.path.exists(all_path.replace('.npy', '.wav')):
            all_path = all_path.replace('.npy', '.wav')
        elif os.path.exists(all_path.replace('.npy', '.mp3')):
            all_path = all_path.replace('.npy', '.mp3')
            
        all_path = Path(all_path)
        y, sr = sf.read(all_path, always_2d=True)
        n_samples = y.shape[0]
        y = y.mean(axis=1)  # mono
        y = torch.Tensor(y)
        
        # resample
        if sr != target_sr:
            y = F.resample(y, sr, target_sr)
            sr = target_sr
            n_samples = y.shape[0]
            
        # pad or crop
        if n_samples < target_frames:
            y = F.pad(y, (0, target_frames - n_samples))
        elif n_samples > target_frames:
            start_idx = random.randint(0, n_samples - target_frames - 1)
            y = y[start_idx: start_idx + target_frames]
            
        return y
            
        
    
class AudioCSVDataModule(LightningDataModule):
    def __init__(self,
                 tokenizer: Tokenizer,
                 dataset_kwargs,
                 dataloader_kwargs,
                 norm_stats: Tuple[float, float],
                 **kwargs):
        super(AudioCSVDataModule, self).__init__()
        self.tokenizer = tokenizer
        self.dataset_kwargs = dataset_kwargs

        # get batch size per device
        devices = dataloader_kwargs.pop("devices", 1)
        if not isinstance(devices, int):
            devices = len(devices)
        batch_size = dataloader_kwargs.pop("batch_size", 256) // devices

        self.dataloader_kwargs = dataloader_kwargs
        self.dataloader_kwargs["batch_size"] = batch_size

        # placeholders
        self.mean, self.std = norm_stats
        
        self.train_dataset = None
        self.val_datasets = None
        self.test_datasets = None

    def setup(self, stage: str | None = None):
        csv_path = self.dataset_kwargs.pop("csv_path", None)
        if csv_path is None:  # already set up, skip
            return
        
        if isinstance(csv_path, ListConfig):
            csv_path = OmegaConf.to_object(csv_path)

        self.train_dataset = AudioCSVDataset(csv_path, **self.dataset_kwargs, split="train", multi_caption="random")

        # for validation and test sets, some options are different  # TODO: handle that shit
        # self.dataset_kwargs["num_frames"] = None
        self.dataset_kwargs["multi_caption"] = "all"

        # we only care about test metrics, fuck validation set
        self.val_datasets = [AudioCSVDataset(p, **self.dataset_kwargs, split="test") for p in csv_path]
        self.test_datasets = [AudioCSVDataset(p, **self.dataset_kwargs, split="test") for p in csv_path]
        # comment this out if needed
        # self.compute_stats()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, collate_fn=collate_audio_text, **self.dataloader_kwargs)

    def val_dataloader(self):
        return [
            DataLoader(dataset, shuffle=False, collate_fn=collate_audio_text, **self.dataloader_kwargs)
            for dataset in self.val_datasets if len(dataset) > 0
        ]
    
    def test_dataloader(self):
        return [
            DataLoader(dataset, shuffle=False, collate_fn=collate_audio_text, **self.dataloader_kwargs)
            for dataset in self.test_datasets if len(dataset) > 0
        ]
    
    def on_before_batch_transfer(
            self, batch: Tuple[List[torch.Tensor], List[str]],
            dataloader_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        audios, descriptions = batch
        # print(descriptions[:3])
        descriptions = [dd for d in descriptions for dd in d]

        # pad audios
        spectrograms = pad_sequence(
            [a.transpose(0, -1) for a in audios],
            batch_first=True,
            padding_value=0
        ).transpose(1, -1)

        # tokenize text inputs
        text_embeddings = self.tokenize(descriptions) if self.tokenizer is not None else torch.zeros(1)
        # print(len(descriptions), spectrograms.shape, len(text_embeddings))
        return spectrograms, text_embeddings, descriptions

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int):
        # normalize spectrogams
        batch[0].sub_(self.mean).div_(self.std)

        return batch

    def tokenize(self, descriptions: List[Any]) -> torch.Tensor:
        return self.tokenizer(descriptions)

    def compute_stats(self, device=None):
        running_stats = OnlineStatsCalculator()
        missing_files = 0

        # do not crop files
        self.dataset.num_frames = 10000

        print("Computing dataset statistics...")
        try:
            for i in trange(len(self.dataset)):
                try:
                    spec = self.dataset[i][0]
                except FileNotFoundError:
                    missing_files += 1
                    print(f"Missing files: {missing_files }")
                    continue
                running_stats.update(spec.mean(dim=-1))
        except Exception as e:
            raise e
        finally:
            print("== Dataset statistics ==")
            print(f"mean: {running_stats.get_mean().cpu().item():.3f}, std: {running_stats.get_std().cpu().item():.3f}")
            print(f"missing_files: {missing_files} over {i + 1} ({100 * missing_files / (i + 1):.2f}%)")
            exit(0)
