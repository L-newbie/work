'''
Author: weidong.he
Date: 2024-05-25 10:57:59
LastEditTime: 2024-05-25 23:54:07
'''
import torch.nn as nn
import torch
from torch.nn import init

#激活函数
class Mish(nn.Module):
    
    def __init__(self):
        super(Mish, self).__init__()
        self.mish = nn.Mish()

    def forward(self, x):
        return self.mish(x)

#CBM模块
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
#残差模块
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None) -> None:
        super(Resblock, self).__init__()
        if hidden_channels is None:
            hidden_channels = channels
        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        x = self.block(x)
        return x
    
#CSP模块
class CSPblock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first) -> None:
        super(CSPblock, self).__init__()
        self.down_sample = BasicConv(in_channels, out_channels, 3, 2, 1)
        #第一个CSP与其他有所差别
        if first:
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                Resblock(out_channels, out_channels//2),
                BasicConv(out_channels, out_channels, 1)
            )
            self.concat_conv = BasicConv(out_channels * 2, out_channels, 1)
        else:
            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)
            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels//2) for _ in range(num_blocks)],
                BasicConv(out_channels//2, out_channels//2, 1)
            )
            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.down_sample(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)
        x = torch.concat([x1, x0], dim=1) 
        x = self.concat_conv(x)
        return x

#backbone
class Net(nn.Module):
    def __init__(self, inplanes = 32, layers = [1, 2, 8, 8, 4], feature_channels = [64, 128, 256, 512, 1024]):
        super(Net, self).__init__()
        self.conv1 = BasicConv(3, inplanes, 3)
        self.stages = nn.ModuleList([
            CSPblock(inplanes, feature_channels[0], layers[0], first=True),
            CSPblock(feature_channels[0], feature_channels[1], layers[1], first=False),
            CSPblock(feature_channels[1], feature_channels[2], layers[2], first=False),
            CSPblock(feature_channels[2], feature_channels[3], layers[3], first=False),
            CSPblock(feature_channels[3], feature_channels[4], layers[4], first=False)
        ])
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight.data, 1)
                init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)
        return out3, out4, out5
    

def load_model_pth(model, pth):
    import numpy as np
    print('Loading weights into state dict, name: %s' % (pth))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pth, map_location=device)
    matched_dict = {}

    for k, v in pretrained_dict.items():
        if np.shape(model_dict[k]) == np.shape(v):
            matched_dict[k] = v
        else:
            print('un matched layers: %s' % k)
    print(len(model_dict.keys()), len(pretrained_dict.keys()))
    print('%d layers matched,  %d layers miss' % (
    len(matched_dict.keys()), len(model_dict) - len(matched_dict.keys())))
    model_dict.update(matched_dict)
    model.load_state_dict(pretrained_dict)
    print('Finished!')
    return model