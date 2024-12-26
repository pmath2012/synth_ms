import numpy as np
from torchvision import transforms
import random
import torch

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
