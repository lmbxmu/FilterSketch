from collections import OrderedDict
import torch.nn as nn
import utils.common as utils

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]

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
        out = nn.AvgPool2d(2)(out)
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
    def __init__(self, sketch_rate, start_conv=1, num_classes=10):
        super(SketchVGG, self).__init__()
        self.sketch_rate = sketch_rate
        self.start_conv = start_conv
        self.current_conv = 0
        self.features = self._make_layers(cfg)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(int(cfg[-2] * sketch_rate[-1]), cfg[-2])),
            ('norm1', nn.BatchNorm1d(cfg[-2])),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(cfg[-2], num_classes)),
        ]))

    def forward(self, x):
        out = self.features(x)
        out = nn.AvgPool2d(2)(out)
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
                if x * self.sketch_rate[self.current_conv - self.start_conv] < in_channels * 3 * 3 \
                        and self.current_conv >= self.start_conv:
                    x = int(x * self.sketch_rate[self.current_conv - self.start_conv])
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
                self.current_conv += 1
        return nn.Sequential(*layers)

class Layerwise_SketchVGG(nn.Module):
    def __init__(self, sketch_rate, sketch_layer=1, num_classes=10):
        super(Layerwise_SketchVGG, self).__init__()
        self.sketch_rate = sketch_rate
        self.sketch_layer = sketch_layer
        self.features = self._make_layers(cfg)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cfg[-2]
                        if sketch_layer != 12 else int(512 * sketch_rate), cfg[-2])),
            ('norm1', nn.BatchNorm1d(cfg[-2])),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(cfg[-2], num_classes)),
        ]))

    def forward(self, x):
        out = self.features(x)
        out = nn.AvgPool2d(2)(out)
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
                if x * self.sketch_rate < in_channels * 3 * 3 and self.sketch_layer != 0:
                    x = int(x * self.sketch_rate)
                    self.sketch_layer -= 1
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

# def test():
#     #python sketch.py --data_set cifar10 --data_path ../data/cifar10 --arch vgg --cfg vgg16 --sketch_model ./experiment/pretrain/vgg_16_bn.pt --job_dir ./experiment/vgg16/sketch --sketch_rate [0.5]*2+[0.3]*2+[0.6]*3+[0.4]*6
#     sketch_rate = '[0.5]*2+[0.3]*2+[0.6]*3+[0.4]*6'
#     sketch_rate = utils.get_sketch_rate(sketch_rate)
#     model = SketchVGG(sketch_rate, start_conv=1)
#     print(model)
#
# test()