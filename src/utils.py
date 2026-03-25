import json
import os
import random

import numpy as np
import torch


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def accuracy(predictions, targets):
    predicted_labels = predictions.argmax(dim=1)
    correct = (predicted_labels == targets).sum().item()
    return correct, targets.size(0)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(path, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
