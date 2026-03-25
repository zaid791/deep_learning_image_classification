import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import yaml

from src.datasets import get_loaders
from src.models import build_model
from src.utils import accuracy, ensure_dir, save_json, seed_everything


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(name, parameters, lr, weight_decay):
    name = name.lower()
    if name == "adam":
        return Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(name, optimizer, epochs):
    name = name.lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.3)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler: {name}")


def mixup_batch(inputs, targets, alpha, device):
    if alpha <= 0:
        return inputs, targets, targets, 1.0

    beta = torch.distributions.Beta(alpha, alpha)
    lam = float(beta.sample().item())
    index = torch.randperm(inputs.size(0), device=device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    return mixed_inputs, targets, targets[index], lam


def cutmix_batch(inputs, targets, alpha, device):
    if alpha <= 0:
        return inputs, targets, targets, 1.0

    beta = torch.distributions.Beta(alpha, alpha)
    lam = float(beta.sample().item())
    index = torch.randperm(inputs.size(0), device=device)

    height = inputs.size(2)
    width = inputs.size(3)
    cut_ratio = torch.sqrt(torch.tensor(1.0 - lam, device=device))
    cut_width = int(width * cut_ratio.item())
    cut_height = int(height * cut_ratio.item())

    center_x = torch.randint(0, width, (1,), device=device).item()
    center_y = torch.randint(0, height, (1,), device=device).item()

    x1 = max(0, center_x - cut_width // 2)
    y1 = max(0, center_y - cut_height // 2)
    x2 = min(width, center_x + cut_width // 2)
    y2 = min(height, center_y + cut_height // 2)

    mixed_inputs = inputs.clone()
    mixed_inputs[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]

    patch_area = (x2 - x1) * (y2 - y1)
    lam_adjusted = 1.0 - patch_area / float(width * height)
    return mixed_inputs, targets, targets[index], lam_adjusted


def soft_target_loss(logits, targets_a, targets_b, lam, criterion):
    return lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)


def train_one_epoch(model, loader, optimizer, criterion, device, augmentation_mode=None, alpha=0.0):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if augmentation_mode == "mixup":
            images, labels_a, labels_b, lam = mixup_batch(images, labels, alpha, device)
        elif augmentation_mode == "cutmix":
            images, labels_a, labels_b, lam = cutmix_batch(images, labels, alpha, device)
        else:
            labels_a = labels
            labels_b = labels
            lam = 1.0

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = soft_target_loss(logits, labels_a, labels_b, lam, criterion)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct, total = accuracy(logits.detach(), labels)
        running_correct += correct
        running_total += total
        batches += 1

    return {
        "loss": running_loss / max(1, batches),
        "accuracy": running_correct / max(1, running_total),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item()
        correct, total = accuracy(logits, labels)
        running_correct += correct
        running_total += total
        batches += 1

    return {
        "loss": running_loss / max(1, batches),
        "accuracy": running_correct / max(1, running_total),
    }


def build_run_name(args):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}_{args.model}_{args.augmentation}_{args.optimizer}_{args.scheduler}"


def load_config(path):
    if not path:
        return {}

    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    return payload or {}


def build_parser(defaults):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=defaults.get("config"))
    parser.add_argument("--data_dir", type=str, default=defaults.get("data_dir", "data/cinic10"))
    parser.add_argument("--model", type=str, default=defaults.get("model", "small_cnn"), choices=["small_cnn", "resnet18", "efficientnet_b0"])
    parser.add_argument("--pretrained", action="store_true", default=defaults.get("pretrained", False))
    parser.add_argument("--epochs", type=int, default=defaults.get("epochs", 1))
    parser.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 64))
    parser.add_argument("--lr", type=float, default=defaults.get("lr", 1e-3))
    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 1e-4))
    parser.add_argument("--dropout", type=float, default=defaults.get("dropout", 0.3))
    parser.add_argument("--optimizer", type=str, default=defaults.get("optimizer", "adam"), choices=["adam", "sgd"])
    parser.add_argument("--scheduler", type=str, default=defaults.get("scheduler", "none"), choices=["none", "cosine", "step"])
    parser.add_argument("--label_smoothing", type=float, default=defaults.get("label_smoothing", 0.0))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--train_fraction", type=float, default=defaults.get("train_fraction", 1.0))
    parser.add_argument("--few_shot_per_class", type=int, default=defaults.get("few_shot_per_class"))
    parser.add_argument("--augmentation", type=str, default=defaults.get("augmentation", "none"), choices=["none", "standard", "strong"])
    parser.add_argument("--advanced_aug", type=str, default=defaults.get("advanced_aug", "none"), choices=["none", "mixup", "cutmix"])
    parser.add_argument("--advanced_aug_alpha", type=float, default=defaults.get("advanced_aug_alpha", 0.0))
    parser.add_argument("--num_workers", type=int, default=defaults.get("num_workers", 0))
    parser.add_argument("--output_dir", type=str, default=defaults.get("output_dir", "runs"))
    parser.add_argument("--smoke_test", action="store_true", default=defaults.get("smoke_test", False))
    return parser


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()

    defaults = load_config(config_args.config)
    parser = build_parser(defaults)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = resolve_device()
    print("Using device:", device)

    train_loader, valid_loader, test_loader = get_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        train_fraction=args.train_fraction,
        few_shot_per_class=args.few_shot_per_class,
        seed=args.seed,
        augmentation=args.augmentation,
        num_workers=args.num_workers,
    )

    if args.smoke_test:
        images, labels = next(iter(train_loader))
        print("Batch images shape:", images.shape)
        print("Batch labels shape:", labels.shape)
        return

    model = build_model(args.model, num_classes=10, dropout=args.dropout, pretrained=args.pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    scheduler = build_scheduler(args.scheduler, optimizer, args.epochs)

    run_name = build_run_name(args)
    run_dir = os.path.join(args.output_dir, run_name)
    ensure_dir(run_dir)

    best_state = None
    best_val_accuracy = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            augmentation_mode=args.advanced_aug,
            alpha=args.advanced_aug_alpha,
        )
        valid_metrics = evaluate(model, valid_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} | train acc {train_metrics['accuracy']:.4f} | "
            f"valid loss {valid_metrics['loss']:.4f} | valid acc {valid_metrics['accuracy']:.4f}"
        )

        if valid_metrics["accuracy"] >= best_val_accuracy:
            best_val_accuracy = valid_metrics["accuracy"]
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "valid_accuracy": best_val_accuracy,
            }
            torch.save(best_state, os.path.join(run_dir, "best_model.pt"))

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")

    payload = {
        "args": vars(args),
        "device": str(device),
        "history": history,
        "best_val_accuracy": best_val_accuracy,
        "test_metrics": test_metrics,
    }
    save_json(os.path.join(run_dir, "metrics.json"), payload)


if __name__ == "__main__":
    main()
