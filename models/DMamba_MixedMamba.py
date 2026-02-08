import torch
import torch.nn as nn
from layers.decomp import DECOMP
from layers.network_mamba_mixed import Network_MixedMamba
from layers.revin import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        c_in = configs.enc_in
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)
        self.ma_type = configs.ma_type
        self.decomp = DECOMP(self.ma_type, configs.alpha, configs.beta)
        self.net = Network_MixedMamba(configs)

    def forward(self, x):
        if self.revin:
            x = self.revin_layer(x, 'norm')
        if self.ma_type == 'reg':
            x = self.net(x, x)
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)
        if self.revin:
            x = self.revin_layer(x, 'denorm')
        return x
