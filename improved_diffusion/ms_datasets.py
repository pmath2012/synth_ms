from PIL import Image
import pandas as pd
import torch
import random
import os
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torchvision.transforms as transforms

LESION_LOAD_MAX = 5471
NUM_LESION_MAX = 25

# Transforms for medical images and masks
class RandomHorizontalFlip:
    def __call__(self, image, mask):
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        return image, mask

class RandomVerticalFlip:
    def __call__(self, image, mask):
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        return image, mask

class RandomRotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        image = transforms.functional.rotate(image, angle)
        mask = transforms.functional.rotate(mask, angle)
        return image, mask

class ToTensorAndNormalize:
    """Convert image and mask to torch tensors. Normalize the image and binarize the mask."""
    def __call__(self, image, mask):
        # Convert to tensor only if not already a tensor
        if not isinstance(image, torch.Tensor):
            image = transforms.functional.to_tensor(image)  # Automatically scales image to [0, 1]
        
        if not isinstance(mask, torch.Tensor):
            mask = transforms.functional.to_tensor(mask)  # This gives a tensor with values between 0 and 1
            mask = torch.where(mask > 0.5, torch.tensor(1.0), torch.tensor(0.0))  # Binarize the mask
        
        return image, mask


class ComposeDouble:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class MSDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.data_frame = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Load image and mask paths
        img_name = os.path.join(self.root_dir, self.data_frame.loc[idx, 'path'])
        label_name = os.path.join(self.root_dir, self.data_frame.loc[idx, 'label'])

        # Load images and masks
        image = Image.open(img_name).convert('L')
        mask = Image.open(label_name).convert('L')

        # Load lesion metadata and normalize it
        lesion_load = float(self.data_frame.loc[idx, 'lesion_load'])
        num_lesions = float(self.data_frame.loc[idx, 'num_lesions'])
        lesion = int(self.data_frame.loc[idx, 'lesion'])

        lesion_load = lesion_load / LESION_LOAD_MAX
        num_lesions = num_lesions / NUM_LESION_MAX

        if self.transform:
            image, mask = self.transform(image, mask)

        # lesion_load = torch.as_tensor(lesion_load, dtype=torch.float32)
        # num_lesions = torch.as_tensor(num_lesions, dtype=torch.float32)
        # lesion = torch.as_tensor(lesion, dtype=torch.float32)
        # Normalize lesion metadata
        lesion_load = lesion_load / LESION_LOAD_MAX
        num_lesions = num_lesions / NUM_LESION_MAX

        # Apply transforms (including the image and mask transformations)
        if self.transform:
            image, mask = self.transform(image, mask)

        # Prepare the conditioning dictionary (out_dict)
        out_dict = {
            "ground_truth": mask,  # Mask tensor
            "lesion_load": torch.tensor(lesion_load, dtype=torch.float32),
            "num_lesions": torch.tensor(num_lesions, dtype=torch.float32),
            "lesion": torch.tensor(lesion, dtype=torch.float32)  # Binary value
        }
        return image, out_dict

# Function to split CSV and create training dataset (without validation)
def load_data(csv_file, root_dir, batch_size, distributed=False, train=True):
    # Load CSV file
    df = pd.read_csv(os.path.join(root_dir, csv_file))
    
    if train:
        transforms = get_train_transforms()
    else:
        transforms = get_test_transforms()
    # Create dataset
    dataset = MSDataset(
        dataframe=df,
        root_dir=root_dir,
        transform=transforms
    )

    if distributed:
        sampler = DistributedSampler(dataset)
        shuffle = False  # DistributedSampler handles the shuffling
    else:
        sampler = None
        shuffle = True

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    #loader = DataLoader(
    #    dataset, 
    #    batch_size=batch_size,
    #    num_workers=1,
    #    pin_memory=True,
    #    drop_last=True,
    #    shuffle=True
    #)
    while True:
        yield from loader

# Get transformations for training
def get_train_transforms():
    return ComposeDouble([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(30),
        ToTensorAndNormalize(),
    ])

def get_test_transforms():
    return ComposeDouble([
        ToTensorAndNormalize(),
    ])

