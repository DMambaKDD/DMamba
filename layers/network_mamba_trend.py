import torch
from torch import nn
from mamba_ssm import Mamba
from layers.Embed import DataEmbedding_inverted
from layers.Mamba_EncDec import Encoder, EncoderLayer

class Network_Mamba_Trend(nn.Module):
    def __init__(self, configs):
        super(Network_Mamba_Trend, self).__init__()

        # Parameters
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.padding_patch = configs.padding_patch

        # --- 1. Non-linear Stream (CNN logic) ---
        self.dim = configs.patch_len * configs.patch_len
        self.patch_num = (configs.seq_len - configs.patch_len) // configs.stride + 1
        if configs.padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, configs.stride)) 
            self.patch_num += 1

        # Patch Embedding
        self.fc1 = nn.Linear(configs.patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)
        
        # CNN Depthwise
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               configs.patch_len, configs.patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream
        self.fc2 = nn.Linear(self.dim, configs.patch_len)

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * configs.patch_len, configs.pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(configs.pred_len * 2, configs.pred_len)

        # --- 2. Linear Stream (Mamba Trend logic) ---
        # Embedding for Trend component
        self.trend_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                       configs.dropout)
        
        # Mamba Encoder for Trend component (Modeling inter-variate dependencies)
        self.trend_encoder = Encoder(
            [
                EncoderLayer(
                        Mamba(
                            d_model=configs.d_model,
                            d_state=configs.d_state,
                            d_conv=2,
                            expand=1,
                        ),
                        Mamba(
                            d_model=configs.d_model,
                            d_state=configs.d_state,
                            d_conv=2,
                            expand=1,
                        ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Project Trend back to pred_len
        self.trend_projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # --- 3. Streams Concatination ---
        self.fc8 = nn.Linear(configs.pred_len * 2, configs.pred_len)

    def forward(self, s, t):
        # s: [Batch, Seq_len, Channel] - seasonality
        # t: [Batch, Seq_len, Channel] - trend
        
        B, L, C = s.shape
        
        # --- Seasonal Stream (CNN) ---
        s_orig = s.permute(0, 2, 1) # [B, C, L]
        s_orig = torch.reshape(s_orig, (B * C, L))
        
        if self.padding_patch == 'end':
            s_orig = self.padding_patch_layer(s_orig)
        s_orig = s_orig.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [B*C, Patch_num, Patch_len]
        
        s_orig = self.fc1(s_orig)
        s_orig = self.gelu1(s_orig)
        s_orig = self.bn1(s_orig)
        
        # [B*C, Patch_num, Dim] -> [B*C, Patch_num, Dim]
        s_res = self.fc2(s_orig) # [B*C, Patch_num, Patch_len]
        
        s_orig = self.conv1(s_orig)
        s_orig = self.gelu2(s_orig)
        s_orig = self.bn2(s_orig)
        
        s_orig = s_orig + s_res
        
        s_orig = self.conv2(s_orig)
        s_orig = self.gelu3(s_orig)
        s_orig = self.bn3(s_orig)
        
        s_orig = self.flatten1(s_orig)
        s_orig = self.fc3(s_orig)
        s_orig = self.gelu4(s_orig)
        s_out = self.fc4(s_orig) # [B*C, S]
        
        # --- Trend Stream (Mamba) ---
        t_enc = self.trend_embedding(t, None) 
        t_enc, _ = self.trend_encoder(t_enc, attn_mask=None)
        t_trend = self.trend_projector(t_enc).permute(0, 2, 1) # [B, S, C]

        # --- Streams Concatination ---
        # Flatten for mixing layer fc8
        s_flat = s_out # [B*C, S]
        t_flat = t_trend.permute(0, 2, 1).reshape(B * C, self.pred_len)
        
        x = torch.cat((s_flat, t_flat), dim=1)
        x = self.fc8(x)
        
        # Final Reshape back to [B, S, C]
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)

        return x
