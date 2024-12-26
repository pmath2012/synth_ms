from torch.utils.data import DataLoader
from .lesions_dataset import MSLesionDataset
from .utils import get_test_transforms, get_train_transforms

def get_train_loaders(train_file,valid_file, data_directory, batch_size):
    train_csv = data_directory+train_file
    valid_csv = data_directory+valid_file
    train_dataset = MSLesionDataset(csv_file=train_csv,
                                    root_dir=data_directory,
                               transform = get_train_transforms())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=32, pin_memory=True)


    valid_dataset = MSLesionDataset(csv_file=valid_csv,
                                    root_dir=data_directory,
                               transform = get_test_transforms())

    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=32, pin_memory=True)

    return train_dataloader, valid_dataloader

def get_synth_loader(train_file, data_directory, batch_size):
    train_csv = data_directory+train_file
    train_dataset = MSLesionDataset(csv_file=train_csv,
                                    root_dir=data_directory,
                               transform = get_train_transforms())

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

    return train_dataloader


def get_test_loader(test_file, data_directory, batch_size):
    test_csv = data_directory+test_file
    test_dataset = MSLesionDataset(csv_file=test_csv,
                                    root_dir=data_directory,
                               transform = get_test_transforms())

    dataloader = DataLoader(test_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    return dataloader
