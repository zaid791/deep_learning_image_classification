import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def _replace_classifier(model, num_classes):
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, num_classes)
        elif hasattr(classifier, "in_features"):
            model.classifier = nn.Linear(classifier.in_features, num_classes)
        else:
            if isinstance(classifier, nn.Sequential):
                last_layer_index = None
                last_linear = None
                for index, layer in enumerate(classifier):
                    if isinstance(layer, nn.Linear):
                        last_layer_index = index
                        last_linear = layer
                if last_layer_index is not None and last_linear is not None:
                    classifier[last_layer_index] = nn.Linear(last_linear.in_features, num_classes)
        return model

    raise ValueError("Unsupported pretrained architecture")


def build_model(name, num_classes=10, dropout=0.3, pretrained=False):
    name = name.lower()

    if name == "small_cnn":
        return SmallCNN(num_classes=num_classes, dropout=dropout)

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        return _replace_classifier(model, num_classes)

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        return _replace_classifier(model, num_classes)

    raise ValueError(f"Unknown model: {name}")
