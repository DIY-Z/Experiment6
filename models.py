import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, in_features, out_features, flag=True):
        """
        in_features: 输入维度长度,类似与nn.Linear中的参数in_features
        out_features: 输出维度长度
        flag: 标记为pre-training还是fine-tuning模式,若为False,则表示fine-tuning,否则表示pre-training
        """
        super(AutoEncoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.flag = flag
        self.encoder = nn.Sequential(
            nn.Linear(self.in_features, self.out_features, bias=True),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.out_features, self.in_features, bias=True), 
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.encoder(x)
        if self.flag:
            return self.decoder(out)
        else:
            return out


class StackedAutoEncoder(nn.Module):
    def __init__(self, layers, flagx=False):
        super(StackedAutoEncoder, self).__init__()
        self.layers = layers
        for layer in self.layers:
            layer.flag = False
        self.encoder1 = self.layers[0]
        self.encoder2 = self.layers[1]
        self.encoder3 = self.layers[2]
        self.encoder4 = self.layers[3]
        self.flagx = flagx
    def forward(self, x):
        out = x.view(x.size(0), -1)
        # out = self.layers[0](out)
        # out = self.layers[1](out)
        # out = self.layers[2](out)
        # out = self.layers[3](out)
        out = self.encoder1(out)
        out = self.encoder2(out)
        if self.flagx:
            return out
        out = self.encoder3(out)
        out = self.encoder4(out)
        # for layer in self.layers:
        #     out = layer(out)
        return out

