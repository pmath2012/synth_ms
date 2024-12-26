import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class RandomHorizontalFlip:
    """Horizontally flip the given image and mask randomly with a probability of 0.5."""
    def __call__(self, image, mask):
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        return image, mask

class RandomVerticalFlip:
    """Vertically flip the given image and mask randomly with a probability of 0.5."""
    def __call__(self, image, mask):
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        return image, mask

class RandomRotation:
    """Rotate the image and mask by a random angle."""
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
        # Normalize image (0-255) to (0-1)
        image = transforms.functional.to_tensor(image)  # Automatically scales image to [0, 1]
        
        # Convert mask to tensor and binarize it
        mask = transforms.functional.to_tensor(mask)  # This gives a tensor with values between 0 and 1
        mask = torch.where(mask > 0.5, torch.tensor(1.0), torch.tensor(0.0))  # Binarize the mask
        
        return image, mask

class ComposeDouble:
    """Compose transforms for both image and mask together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class MSDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the csv file with annotations (path and label).
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)
 
    def __getitem__(self, idx):
        # Fetch the image and mask paths using the column names 'path' and 'label'
        img_name = os.path.join(self.root_dir, 'image', self.data_frame.loc[idx, 'path'])
        label_name = os.path.join(self.root_dir, 'label', self.data_frame.loc[idx, 'label'])

        image = Image.open(img_name).convert('L')  # Load the image as RGB
        mask = Image.open(label_name).convert('L')  # Load the mask as grayscale

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

# Example of how to use MSDataset with transformations
def get_train_transforms():
    return ComposeDouble([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(30),  # Rotate the image by a random angle up to 30 degrees
        ToTensorAndNormalize(),  # Convert the PIL Image to a tensor, normalize, and binarize the mask
    ])

def get_val_transforms():
    return ComposeDouble([
        ToTensorAndNormalize(),  # No augmentation for validation, just convert to tensor, normalize, and binarize
    ])