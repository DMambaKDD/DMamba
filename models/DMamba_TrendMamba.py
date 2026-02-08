import torch
import torch.nn as nn
from layers.decomp import DECOMP
from layers.revin import RevIN
from models.DMamba_T import Model as TMamba
import copy

class Network_TrendMamba(nn.Module):
    def __init__(self, configs):
        super(Network_TrendMamba, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.padding_patch = configs.padding_patch
        
        self.dim = configs.patch_len * configs.patch_len
        self.patch_num = (configs.seq_len - configs.patch_len) // configs.stride + 1
        if configs.padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, configs.stride)) 
            self.patch_num += 1

        self.fc1 = nn.Linear(configs.patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num, configs.patch_len, configs.patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)
        self.fc2 = nn.Linear(self.dim, configs.patch_len)
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * configs.patch_len, configs.pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(configs.pred_len * 2, configs.pred_len)

        mamba_configs = copy.deepcopy(configs)
        mamba_configs.enc_in = 1
        self.trend_mamba = TMamba(mamba_configs)
        self.fc8 = nn.Linear(configs.pred_len * 2, configs.pred_len)

    def forward(self, s, t):
        B, L, C = s.shape
        s = s.permute(0, 2, 1).reshape(B * C, L)
        t = t.permute(0, 2, 1).reshape(B * C, L)

        if self.padding_patch == 'end':
            s_in = self.padding_patch_layer(s.unsqueeze(1)).squeeze(1)
        else:
            s_in = s
        s_patch = s_in.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        s_out = self.fc1(s_patch)
        s_out = self.gelu1(s_out)
        s_out = self.bn1(s_out)
        res = s_out
        s_out = self.conv1(s_out)
        s_out = self.gelu2(s_out)
        s_out = self.bn2(s_out)
        res = self.fc2(res)
        s_out = s_out + res
        s_out = self.conv2(s_out)
        s_out = self.gelu3(s_out)
        s_out = self.bn3(s_out)
        s_out = self.flatten1(s_out)
        s_out = self.fc3(s_out)
        s_out = self.gelu4(s_out)
        s_out = self.fc4(s_out)

        t_in = t.unsqueeze(-1)
        t_out = self.trend_mamba(t_in, None, None, None).squeeze(-1)

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
        self.net = Network_TrendMamba(configs)

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
