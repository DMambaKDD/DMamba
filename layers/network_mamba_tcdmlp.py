import torch
from torch import nn
from mamba_ssm import Mamba
from layers.Embed import DataEmbedding_inverted
from layers.Mamba_EncDec import Encoder, EncoderLayer

class Network_Mamba_TCDMLP(nn.Module):
    def __init__(self, configs):
        super(Network_Mamba_TCDMLP, self).__init__()

        # Parameters
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in

        # --- 1. Seasonal Stream (Mamba) ---
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

        # --- 2. Trend Stream (Channel-Dependent MLP) ---
        # Instead of CI-MLP, we use a CD-MLP architecture
        # Step 1: Temporal projection (CI)
        self.t_temporal_fc = nn.Linear(configs.seq_len, configs.pred_len)
        # Step 2: Channel mixing (CD)
        self.t_channel_fc = nn.Linear(configs.enc_in, configs.enc_in)
        
        self.t_gelu = nn.GELU()
        self.t_dropout = nn.Dropout(configs.dropout)
        self.t_norm = nn.LayerNorm(configs.enc_in)

        # --- 3. Streams Concatenation ---
        self.fc8 = nn.Linear(configs.pred_len * 2, configs.pred_len)

    def forward(self, s, t):
        # s: [Batch, Seq_len, Channel] - seasonality
        # t: [Batch, Seq_len, Channel] - trend
        
        B, L, C = s.shape
        
        # --- Seasonal Stream (Mamba) ---
        s_enc = self.s_embedding(s, None) 
        s_enc, _ = self.s_encoder(s_enc, attn_mask=None)
        s_out = self.s_projector(s_enc).permute(0, 2, 1) # [B, S, C]

        # --- Trend Stream (Channel-Dependent MLP) ---
        # t: [B, L, C] -> [B, C, L]
        t_work = t.permute(0, 2, 1)
        # Temporal projection: [B, C, L] -> [B, C, S]
        t_work = self.t_temporal_fc(t_work)
        t_work = self.t_gelu(t_work)
        
        # Channel mixing: [B, C, S] -> [B, S, C]
        t_work = t_work.permute(0, 2, 1)
        # [B, S, C] -> [B, S, C] (Mixing across C dimension)
        t_trend = self.t_channel_fc(t_work)
        t_trend = self.t_norm(t_trend)
        t_trend = self.t_dropout(t_trend) # [B, S, C]

        # --- Streams Concatenation ---
        # Flatten for mixing layer fc8
        s_flat = s_out.permute(0, 2, 1).reshape(B * C, self.pred_len)
        t_flat = t_trend.permute(0, 2, 1).reshape(B * C, self.pred_len)
        
        x = torch.cat((s_flat, t_flat), dim=1)
        x = self.fc8(x)
        
        # Final Reshape back to [B, S, C]
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)

        return x
