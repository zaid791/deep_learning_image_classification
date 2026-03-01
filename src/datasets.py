import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CINIC_MEAN = [0.47889522, 0.47227842, 0.43047404]
CINIC_STD  = [0.24205776, 0.23828046, 0.25874835]

def get_loaders(data_dir, batch_size=64):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CINIC_MEAN, CINIC_STD),
    ])

    train_ds = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=transform
    )

    valid_ds = datasets.ImageFolder(
        os.path.join(data_dir, "valid"),
        transform=transform
    )

    test_ds = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform=transform
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, valid_loader, test_loader