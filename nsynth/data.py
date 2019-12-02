import json
import random
from glob import glob
from os import path
from typing import Optional, List, Dict

import librosa
import torch
from torch import dtype as torch_dtype
from torch.utils import data

from .functional import encode_μ_law


class NSynthDataset(data.Dataset):
    """
    Dataset to handle the NSynth data in json/wav format.
    """

    def __init__(self, root: str, subset: str = 'train',
                 families: Optional[List[str]] = None,
                 sources: Optional[List[str]] = None,
                 dtype: torch_dtype = torch.float32, mono: bool = True):
        """
        :param root: The path to the dataset. Should contain the sub-folders
            for the splits as extracted from the .tar.gz.
        :param subset: The subset to use.
        :param families: Only keep those Instrument families
        :param sources: Only keep those instrument sources
        :param dtype: The data type to output for the audio signals.
        :param mono: Whether to use mono signal instead of stereo.
        """
        self.dtype = dtype
        self.subset = subset.lower()
        self.mono = mono

        if isinstance(families, str):
            families = [families]
        if isinstance(sources, str):
            sources = [sources]

        assert self.subset in ['valid', 'test', 'train']

        self.root = path.normpath(f'{root}/nsynth-{subset}')
        if not path.isdir(self.root):
            raise ValueError('The given root path is not a directory.'
                             f'\nI got {self.root}')

        if not path.isfile(f'{self.root}/examples.json'):
            raise ValueError('The given root path does not contain an'
                             'examples.json')

        print(f'Loading NSynth data from split {self.subset} at {self.root}')

        with open(f'{self.root}/examples.json', 'r') as fp:
            self.attrs = json.load(fp)

        if families:
            self.attrs = {k: a for k, a in self.attrs.items()
                          if a['instrument_family_str'] in families}
        if sources:
            self.attrs = {k: a for k, a in self.attrs.items()
                          if a['instrument_source_str'] in sources}

        print(f'\tFound {len(self)} samples.')

        files_on_disk = set(map(lambda x: path.basename(x)[:-4],
                                glob(f'{self.root}/audio/*.wav')))
        if not set(self.attrs) <= files_on_disk:
            raise FileNotFoundError

        self.names = list(self.attrs.keys())

    def __len__(self):
        return len(self.attrs)

    def __str__(self):
        return f'NSynthDataset: {len(self):>7} samples in subset {self.subset}'

    def __getitem__(self, item: int):
        name = self.names[item]
        attrs = self.attrs[name]
        path = f'{self.root}/audio/{name}.wav'
        raw, _ = librosa.load(path, mono=self.mono, sr=attrs['sample_rate'])
        # Add channel dimension.
        if raw.ndim == 1:
            raw = raw[None, ...]
        attrs['audio'] = torch.tensor(raw, dtype=self.dtype)
        return attrs


class AudioOnlyNSynthDataset(NSynthDataset):
    def __init__(self, *args, crop: Optional[int] = None, **kwargs):
        super(AudioOnlyNSynthDataset, self).__init__(*args, **kwargs)
        self.crop = crop

    def __getitem__(self, item: int):
        attrs = super(AudioOnlyNSynthDataset, self).__getitem__(item)
        # Make a random crop if given
        if self.crop:
            pivot = random.randint(0, attrs['audio'].shape[1] - self.crop)
            attrs['audio'] = attrs['audio'][:, pivot:pivot + self.crop]
        # μ-Law gives us range [-128, 128]
        # Input is in range [-1, 1]
        # The loss CrossEntropy Targets are range [0, 255]
        audio = encode_μ_law(attrs['audio'])
        audio_scaled = audio / 128
        audio_target = (audio.squeeze() + 128).long()
        return audio_scaled, audio_target


def make_loaders(data_dir: str, subsets: List[str], nbatch: int,
                 crop: int = 6144, families: Optional[List[str]] = None,
                 sources: Optional[List[str]] = None) \
        -> Dict[str, data.DataLoader]:
    """
    Make a dictionary of data loaders for the given subsets.
    :param data_dir: Location of the Dataset
    :param subsets: subsets to make datasets for
    :param nbatch: batch size
    :param crop: length  of the cropped samples
    :param families: families of instruments to select
    :param sources: sources of instruments to select
    :return:
    """
    data_loaders = dict()
    for subset in subsets:
        dset = AudioOnlyNSynthDataset(root=data_dir, subset=subset,
                                      families=families, sources=sources,
                                      crop=crop)
        data_loaders[subset] = data.DataLoader(
            dset, batch_size=nbatch, num_workers=8, shuffle=True)
    return data_loaders
