import torch
import torch.nn as nn
from layers.decomp import DECOMP
from layers.network_mamba import Network_Mamba
from layers.revin import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # Parameters
        c_in = configs.enc_in       # input channels

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)

        # Moving Average (Decomposition)
        self.ma_type = configs.ma_type
        alpha = configs.alpha       # smoothing factor for EMA
        beta = configs.beta         # smoothing factor for DEMA

        self.decomp = DECOMP(self.ma_type, alpha, beta)
        
        # New Network with Mamba seasonal stream
        self.net = Network_Mamba(configs)

    def forward(self, x):
        # x: [Batch, Input_len, Channel]

        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.ma_type == 'reg':   # If no decomposition
            x = self.net(x, x)
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)

        # Denormalization
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x
