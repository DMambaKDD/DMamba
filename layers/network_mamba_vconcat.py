import torch
from torch import nn
from mamba_ssm import Mamba
from layers.Embed import DataEmbedding_inverted
from layers.Mamba_EncDec import Encoder, EncoderLayer

class Network_Mamba_VConcat(nn.Module):
    def __init__(self, configs):
        super(Network_Mamba_VConcat, self).__init__()

        # Parameters
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in

        # --- Seasonal Stream (Mamba) ---
        self.s_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        self.s_encoder = Encoder(
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
        
        self.s_projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # --- Trend Stream (Channel-Independent MLP) ---
        self.t_fc5 = nn.Linear(configs.seq_len, configs.pred_len * 4)
        self.t_avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.t_ln1 = nn.LayerNorm(configs.pred_len * 2)

        self.t_fc6 = nn.Linear(configs.pred_len * 2, configs.pred_len)
        self.t_avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.t_ln2 = nn.LayerNorm(configs.pred_len // 2)

        self.t_fc7 = nn.Linear(configs.pred_len // 2, configs.pred_len)

        # --- Variable Dimension Fusion (Channel Concatenation) ---
        self.fc_fusion = nn.Linear(configs.enc_in * 2, configs.enc_in)

    def forward(self, s, t):
        # s: [Batch, Seq_len, Channel] - seasonality
        # t: [Batch, Seq_len, Channel] - trend
        
        B, L, C = s.shape
        
        # 1. Seasonal Stream (Mamba)
        s_enc = self.s_embedding(s, None) 
        s_enc, _ = self.s_encoder(s_enc, attn_mask=None)
        # [B, C, d_model] -> [B, C, S] -> [B, S, C]
        s_out = self.s_projector(s_enc).permute(0, 2, 1)

        # 2. Trend Stream (MLP)
        # t: [B, L, C] -> [B, C, L] -> [B*C, L]
        t_flat = t.permute(0, 2, 1).reshape(B * C, L)
        
        t_flat = self.t_fc5(t_flat)
        t_flat = self.t_avgpool1(t_flat)
        t_flat = self.t_ln1(t_flat)

        t_flat = self.t_fc6(t_flat)
        t_flat = self.t_avgpool2(t_flat)
        t_flat = self.t_ln2(t_flat)

        t_flat = self.t_fc7(t_flat) # [B*C, S]
        
        # Reshape back to [B, C, S] -> [B, S, C]
        t_out = t_flat.reshape(B, C, self.pred_len).permute(0, 2, 1)

        # 3. Variable Dimension Fusion (Channel Concatenation)
        # s_out: [B, S, C]
        # t_out: [B, S, C]
        # Concatenate along the channel dimension (dim=2)
        combined = torch.cat((s_out, t_out), dim=2) # [B, S, 2*C]
        
        # Project back to original channel dimension C
        out = self.fc_fusion(combined) # [B, S, C]

        return out
