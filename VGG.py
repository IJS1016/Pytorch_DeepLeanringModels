import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class VGG16(nn.Module) : 
    def __init__(self, n_classes):
        super(VGG16, self).__init__()
        config = [64, 64, "P", 128, 128, "P", 256, 256, "P", 512, 512, 512, "P", 512, 512, 512, "P"]
        self.feature_extractor = self.make_layers(config)

        # for init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def make_layers(self, config):
        layers = []
        in_channels = 3
        for i in config:
            if i == "P" : 
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            else :
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=i, kernel_size=3, stride=1, padding=1)]
                layers += [nn.ReLU()]
                in_channels = i

        block = nn.Sequential(*layers)

        return block



