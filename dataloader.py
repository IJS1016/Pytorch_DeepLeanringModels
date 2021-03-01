import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


class ImageDataset(Dataset):
    def __init__(self, data_path, transforms_=None, mode='train'):
        self.transform = transforms_
        self.dir = sorted(glob.glob(os.path.join(data_path,'%s' % mode)+'/*'))
        self.data = []
        for i, dirs in enumerate(self.dir):
            self.data.append((sorted(glob.glob(dirs+'/*')), i ))

    def __getitem__(self, index):
        img = Image.open(self.data[index % len(self.data)])

        img = self.transform(img)
        
        return img

    def __len__(self):
        return max(len(self.files))