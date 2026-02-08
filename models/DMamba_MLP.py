import torch
import torch.nn as nn
from layers.decomp import DECOMP
from layers.network_mlp import NetworkMLP
from layers.revin import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        c_in = configs.enc_in
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)
        self.ma_type = configs.ma_type
        self.decomp = DECOMP(self.ma_type, configs.alpha, configs.beta)
        self.net_s = NetworkMLP(configs.seq_len, configs.pred_len)
        self.net_t = NetworkMLP(configs.seq_len, configs.pred_len)

    def forward(self, x):
        if self.revin:
            x = self.revin_layer(x, 'norm')
        if self.ma_type == 'reg':
            x = self.net_s(x)
        else:
            seasonal_init, trend_init = self.decomp(x)
            out_s = self.net_s(seasonal_init)
            out_t = self.net_t(trend_init)
            x = out_s + out_t
        if self.revin:
            x = self.revin_layer(x, 'denorm')
        return x
