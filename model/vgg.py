from collections import OrderedDict
import torch.nn as nn

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cfg[-2], cfg[-2])),
            ('norm1', nn.BatchNorm1d(cfg[-2])),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(cfg[-2], num_classes)),
        ]))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

class SketchVGG(nn.Module):
    def __init__(self, sketch_rate, num_classes=10):
        super(SketchVGG, self).__init__()
        self.sketch_rate = sketch_rate
        self.features = self._make_layers(cfg)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(int(cfg[-2] * sketch_rate), cfg[-2])),
            ('norm1', nn.BatchNorm1d(cfg[-2])),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(cfg[-2], num_classes)),
        ]))

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if x * self.sketch_rate < in_channels * 3 * 3:
                    x = int(x * self.sketch_rate)
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)
