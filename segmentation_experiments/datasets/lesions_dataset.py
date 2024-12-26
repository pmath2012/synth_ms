import torch
import  os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class MSLesionDataset(Dataset):
    """MS lesion dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ms_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ms_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, 'image', self.ms_frame.loc[idx, 'path'])
        mask_name = os.path.join(self.root_dir, 'label', self.ms_frame.loc[idx, 'label'])
        image = Image.open(img_name).convert('L')
        mask = Image.open(mask_name).convert('L')

        if self.transform:
            image, mask = self.transform(image, mask)

        sample = {'image': image, 'mask': mask.long()}

        return sample