import torch
import torch.nn as nn
import torchvision.models as models

class CXRModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features,
            num_labels
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))
