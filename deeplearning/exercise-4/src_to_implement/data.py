from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import pandas

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data: pandas.DataFrame, mode: str):

        if mode == 'train':
            self._transform: tv.transforms.Compose = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        else:
            self._transform: tv.transforms.Compose = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = self.data.iloc[index, 0]
        label = self.data.iloc[index, 1]
        image = imread(img_name)
        image = gray2rgb(image)

        if self._transform:
            image = self._transform(image)

        if label == 1:
            label = torch.tensor([0, 1])
        else:
            label = torch.tensor([1, 0])
        return image, label


