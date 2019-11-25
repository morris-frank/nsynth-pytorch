import json
import os
import random
from glob import glob

import librosa
import torch
from torch import dtype as torch_dtype
from torch.utils import data


class NSynthDataset(data.Dataset):
    def __init__(self,
                 root: str = '/home/morris/var/data/nsynth/test/',
                 subset: str = 'train',
                 dtype: torch_dtype = torch.float32,
                 mono: bool = True):
        self.dtype = dtype
        self.subset = subset
        self.mono = mono

        self.root = os.path.normpath(root)
        if not os.path.isdir(self.root):
            raise ValueError('The given root path is not a directory.')

        if not os.path.isfile(f'{self.root}/examples.json'):
            raise ValueError('The given root path does not contain an'
                             'examples.json')

        with open(f'{self.root}/examples.json', 'r') as fp:
            self.attrs = json.load(fp)

        for file in glob(f'{self.root}/audio/*.wav'):
            assert os.path.basename(file)[:-4] in self.attrs

        self.names = list(self.attrs.keys())

    def __len__(self):
        return len(self.attrs)

    def __getitem__(self, item: int):
        name = random.choice(self.names)
        attrs = self.attrs[name]
        path = f'{self.root}/audio/{name}.wav'
        raw, _ = librosa.load(path, mono=self.mono, sr=attrs['sample_rate'])
        attrs['audio'] = raw
        return attrs
