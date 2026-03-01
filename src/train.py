import argparse
import torch

from src.datasets import get_loaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/cinic10")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, valid_loader, test_loader = get_loaders(args.data_dir)

    # Just test one batch
    images, labels = next(iter(train_loader))

    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)


if __name__ == "__main__":
    main()