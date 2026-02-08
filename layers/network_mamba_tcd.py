import torch
from torch import nn
from mamba_ssm import Mamba
from layers.Embed import DataEmbedding_inverted
from layers.Mamba_EncDec import Encoder, EncoderLayer

class Network_Mamba_TCD(nn.Module):
    def __init__(self, configs):
        super(Network_Mamba_TCD, self).__init__()

        # Parameters
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in

        # --- 1. Seasonal Stream (Mamba - Channel Dependent) ---
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

        # --- 2. Trend Stream (Mamba - Channel Dependent) ---
        self.t_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        self.t_encoder = Encoder(
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
        
        self.t_projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # --- 3. Streams Concatenation ---
        self.fc8 = nn.Linear(configs.pred_len * 2, configs.pred_len)

    def forward(self, s, t):
        B, L, C = s.shape
        
        # --- Seasonal Stream ---
        s_enc = self.s_embedding(s, None) 
        s_enc, _ = self.s_encoder(s_enc, attn_mask=None)
        s_out = self.s_projector(s_enc).permute(0, 2, 1)

        # --- Trend Stream ---
        t_enc = self.t_embedding(t, None) 
        t_enc, _ = self.t_encoder(t_enc, attn_mask=None)
        t_out = self.t_projector(t_enc).permute(0, 2, 1)

        # --- Streams Concatenation ---
        s_flat = s_out.permute(0, 2, 1).reshape(B * C, self.pred_len)
        t_flat = t_out.permute(0, 2, 1).reshape(B * C, self.pred_len)
        
        x = torch.cat((s_flat, t_flat), dim=1)
        x = self.fc8(x)
        
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)
        return x
