import torch
import torch.nn as nn
from layers.decomp import DECOMP
from layers.network_mamba_hybrid import Network_Mamba_Hybrid
from layers.revin import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.revin = configs.revin
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        self.ma_type = configs.ma_type
        self.decomp = DECOMP(self.ma_type, configs.alpha, configs.beta)
        
        # In hybrid model, we use fixed 96 for seasonal lookback and configs.seq_len for trend
        self.seasonal_seq_len = 96
        self.trend_seq_len = configs.seq_len
        
        self.net = Network_Mamba_Hybrid(configs, self.seasonal_seq_len, self.trend_seq_len)

    def forward(self, x):
        if self.revin:
            x = self.revin_layer(x, 'norm')
            
        seasonal_init, trend_init = self.decomp(x)
        
        # Crop seasonal_init to seasonal_seq_len (96)
        if seasonal_init.shape[1] > self.seasonal_seq_len:
            seasonal_init = seasonal_init[:, -self.seasonal_seq_len:, :]
            
        x = self.net(seasonal_init, trend_init)
        
        if self.revin:
            x = self.revin_layer(x, 'denorm')
        return x
