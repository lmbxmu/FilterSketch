import torch
import torch.nn as nn
import torch.nn.functional as F

norm_mean, norm_var = 0.0, 1.0

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, sketch_rate=1.0):
        super(ResBasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        middle_planes = int(sketch_rate * planes)
        self.conv1 = conv3x3(inplanes, middle_planes, stride)
        self.bn1 = nn.BatchNorm2d(middle_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(middle_planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_layers, sketch_rate=1.0, num_classes=10):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6

        self.sketch_rate = sketch_rate
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, blocks=n, stride=1)
        self.layer2 = self._make_layer(block, 32, blocks=n, stride=2)
        self.layer3 = self._make_layer(block, 64, blocks=n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        layers = []
        layers.append(block(self.inplanes, planes, stride, sketch_rate=self.sketch_rate))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, sketch_rate=self.sketch_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet56(**kwargs):
    return ResNet(ResBasicBlock, 56, **kwargs)

def resnet110(**kwargs):
    return ResNet(ResBasicBlock, 110, **kwargs)

def test():
    model = resnet56(sketch_rate=0.5)
    y = model(torch.randn(1, 3, 32, 32))
    print(y.size())
    print(model)
    # ckpt = torch.load('../checkpoint/resnet_56.pt', map_location='cpu')
    # state_dict = ckpt['state_dict']
    # model.load_state_dict(state_dict)
    # for k, v in model.state_dict().items():
    #     print(k, v.size())

# test()