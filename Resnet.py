import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class Resnet18(nn.Module) :
    def __init__(self, n_classes) :
        super(Resnet18, self).__init__()

        fill_mode = 'zero_padding'
        self.n_classes = n_classes
        
        in_ch     = 64
        config    = [[3, 64], [1, 128, fill_mode], [1, 128], [1, 256, fill_mode], [1, 256], [1, 512, fill_mode], [1, 512]]

        layers = []

        layers += [nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=2),
                                              nn.BatchNorm2d(64),
                                              nn.ReLU(),
                                              nn.MaxPool2d(kernel_size=3, stride=2)]

        layers += make_layers(config, in_ch)
        self.model = nn.Sequential(*layers)


    def forward(self, x) :
        x = self.model(x)


        x = nn.AvgPool2d(kernel_size=1)(x) # for CIFAR10
        
        #x = nn.AvgPool2d(kernel_size=7)(x) # for imagenet
        
        x = torch.flatten(x, 1)
        probs = F.softmax(x, dim=1)

        return x, probs


class Resnet34(nn.Module) :
    def __init__(self, n_classes) :
        super(Resnet34, self).__init__()

        fill_mode = 'zero_padding'
        self.n_classes = n_classes
        
        in_ch  = 64
        config    = [[3, 64], [1, 128, fill_mode], [3, 128], [1, 256, fill_mode], [5, 256], [1, 512, fill_mode], [2, 512]]

        layers = []

        layers += [nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=2),
                                              nn.BatchNorm2d(64),
                                              nn.ReLU(),
                                              nn.MaxPool2d(kernel_size=3, stride=2)]

        layers += make_layers(config, in_ch)
        self.model = nn.Sequential(*layers)


    def forward(self, x) :
        x = self.model(x)


        x = nn.AvgPool2d(kernel_size=1)(x) # for CIFAR10
        
        # x = nn.AvgPool2d(kernel_size=7)(x) # for imagenet
        
        x = torch.flatten(x, 1)
        probs = F.softmax(x, dim=1)

        return x, probs

def make_layers(config, in_ch) :
    layers = []
    for bb in config :
        block_n = bb[0]
        out_ch = bb[1]

        stride = 2 if len(bb) == 3 else 1

        layers += big_block(in_ch, out_ch, stride, block_n).model
        in_ch   = out_ch
    
    return layers


class big_block(nn.Module) :
    def __init__(self, in_ch, out_ch, stride, block_n) :
        super(big_block, self).__init__()
        layers = []
        for _ in range(block_n) :
            layers += block(in_ch, out_ch, stride).model
            in_ch = out_ch
            stride = 1

        self.model = nn.Sequential(*layers)

    
    def forward(self, x) :
        return self.model(x)


class block(nn.Module) :
    def __init__(self, in_ch, out_ch, stride=1) :
        super(block, self).__init__()
        self.model = nn.Sequential(conv_3x3(in_ch, out_ch, stride),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(),
                                   conv_3x3(out_ch, out_ch),
                                   nn.BatchNorm2d(out_ch))
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1,padding=0)
    def forward(self, x) :
        short_cut = self.conv1(x)
        x  = self.model(x)

        x += short_cut
        x  = nn.ReLU()(x)

        return x


def conv_3x3(in_ch, out_ch, stride=1) :
    return nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1)




###################################################################################
# plain net for comparing
###################################################################################
class Plain18(nn.Module) : 
    """plain for comparing Resnet18"""
    def __init__(self, n_classes):
        super(plain, self).__init__()
        config = [64, 64, 64, 128, 128, 256, 256, 512, 512]
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


