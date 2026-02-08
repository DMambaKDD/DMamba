import torch
import torch.nn as nn
from layers.decomp import DECOMP
from layers.revin import RevIN
from models.DMamba_T import Model as TMamba
import copy

class Network_DualMamba(nn.Module):
    def __init__(self, configs):
        super(Network_DualMamba, self).__init__()
        self.pred_len = configs.pred_len
        
        # --- Seasonal Stream (DMamba_T) ---
        mamba_configs_s = copy.deepcopy(configs)
        mamba_configs_s.enc_in = 1
        self.seasonal_mamba = TMamba(mamba_configs_s)

        # --- Trend Stream (DMamba_T) ---
        mamba_configs_t = copy.deepcopy(configs)
        mamba_configs_t.enc_in = 1
        self.trend_mamba = TMamba(mamba_configs_t)

        # --- Mixing ---
        self.fc8 = nn.Linear(configs.pred_len * 2, configs.pred_len)

    def forward(self, s, t):
        B, L, C = s.shape
        s_ci = s.permute(0, 2, 1).reshape(B * C, L)
        t_ci = t.permute(0, 2, 1).reshape(B * C, L)

        s_in = s_ci.unsqueeze(-1)
        s_out = self.seasonal_mamba(s_in, None, None, None) 
        s_out = s_out.squeeze(-1)

        t_in = t_ci.unsqueeze(-1)
        t_out = self.trend_mamba(t_in, None, None, None)
        t_out = t_out.squeeze(-1)

        x = torch.cat((s_out, t_out), dim=1)
        x = self.fc8(x)
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.revin = configs.revin
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        self.ma_type = configs.ma_type
        self.decomp = DECOMP(self.ma_type, configs.alpha, configs.beta)
        self.net = Network_DualMamba(configs)

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
