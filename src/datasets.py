import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CINIC_MEAN = [0.47889522, 0.47227842, 0.43047404]
CINIC_STD = [0.24205776, 0.23828046, 0.25874835]


def build_train_transform(augmentation=None):
    ops = []

    if augmentation == "standard":
        ops.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        )
    elif augmentation == "strong":
        ops.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
            ]
        )

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CINIC_MEAN, CINIC_STD),
        ]
    )
    return transforms.Compose(ops)


def build_eval_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CINIC_MEAN, CINIC_STD),
        ]
    )


def _build_dataset(data_dir, split, transform):
    return datasets.ImageFolder(os.path.join(data_dir, split), transform=transform)


def _subset_by_fraction(dataset, fraction=None, seed=42):
    if fraction is None or fraction >= 1.0:
        return dataset

    fraction = max(0.0, min(1.0, fraction))
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    candidate_indices = dataset.indices if isinstance(dataset, Subset) else range(len(base_dataset))

    class_to_indices = {}
    for index in candidate_indices:
        _, label = base_dataset.samples[index]
        class_to_indices.setdefault(label, []).append(index)

    generator = torch.Generator().manual_seed(seed)
    selected = []
    for label in sorted(class_to_indices):
        indices = class_to_indices[label]
        order = torch.randperm(len(indices), generator=generator).tolist()
        keep = max(1, int(len(indices) * fraction))
        selected.extend(indices[position] for position in order[:keep])

    selected.sort()
    return Subset(base_dataset, selected)


def _few_shot_subset(dataset, shots_per_class, seed=42):
    if shots_per_class is None:
        return dataset

    class_to_indices = {}
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    candidate_indices = dataset.indices if isinstance(dataset, Subset) else range(len(dataset))

    for index in candidate_indices:
        _, label = base_dataset.samples[index]
        class_to_indices.setdefault(label, []).append(index)

    generator = torch.Generator().manual_seed(seed)
    selected = []
    for label in sorted(class_to_indices):
        indices = class_to_indices[label]
        order = torch.randperm(len(indices), generator=generator).tolist()
        selected.extend(indices[position] for position in order[: max(1, shots_per_class)])

    selected.sort()
    return Subset(base_dataset, selected)


def get_loaders(
    data_dir,
    batch_size=64,
    train_fraction=1.0,
    few_shot_per_class=None,
    seed=42,
    augmentation="none",
    num_workers=0,
):

    train_transform = build_train_transform(augmentation=augmentation)
    eval_transform = build_eval_transform()

    train_ds = _build_dataset(data_dir, "train", train_transform)
    valid_ds = _build_dataset(data_dir, "valid", eval_transform)
    test_ds = _build_dataset(data_dir, "test", eval_transform)

    train_ds = _subset_by_fraction(train_ds, fraction=train_fraction, seed=seed)
    train_ds = _few_shot_subset(train_ds, shots_per_class=few_shot_per_class, seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, valid_loader, test_loader
