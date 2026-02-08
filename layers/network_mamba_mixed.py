import torch
from torch import nn
from mamba_ssm import Mamba
from layers.Embed import DataEmbedding_inverted, DataEmbedding
from layers.Mamba_EncDec import Encoder, EncoderLayer

class Network_MixedMamba(nn.Module):
    def __init__(self, configs):
        super(Network_MixedMamba, self).__init__()

        # Parameters
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in

        # --- 1. Seasonal Stream (Mamba logic - Variable Independence) ---
        # Embedding: Projects the time dimension of each variate into d_model
        # [B, L, C] -> [B, C, L] -> [B, C, E]
        self.s_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # Mamba Encoder: Models dependencies between variables
        # Input: [B, C, E] (Channel becomes Sequence)
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
        
        # Projector: [B, C, E] -> [B, C, S]
        self.s_projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # --- 2. Trend Stream (Mamba logic - Time Dependence) ---
        # Embedding: Projects variables into d_model, preserving time
        # [B, L, C] -> [B, L, E]
        self.t_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        
        # Mamba Encoder: Models time dependencies
        # Input: [B, L, E] (Time is Sequence)
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
        
        # Projector for Trend stream
        # 1. Time Projection: [B, L, E] -> [B, S, E]
        self.t_time_projector = nn.Linear(configs.seq_len, configs.pred_len, bias=True)
        # 2. Feature Projection: [B, S, E] -> [B, S, C]
        self.t_feature_projector = nn.Linear(configs.d_model, configs.enc_in, bias=True)

        # --- 3. Streams Concatenation ---
        # Both streams output [B, S, C]
        # Seasonal stream outputs [B, C, S] -> [B, S, C] (Mixing variables in Mamba encoder).
        
        # Align shapes for fusion
        # Seasonal stream output s_out is [B, S, C]
        # Trend stream output t_out is [B, S, C]
        
        # We can concatenate along a new dimension or feature dimension and project back?
        # Original DMamba logic:
        # s_flat = s_out.permute(0, 2, 1).reshape(B * C, self.pred_len)
        # t_flat = t_out.permute(0, 2, 1).reshape(B * C, self.pred_len)
        # x = torch.cat((s_flat, t_flat), dim=1)
        # x = self.fc8(x)
        
        self.fc8 = nn.Linear(configs.pred_len * 2, configs.pred_len)

    def forward(self, s, t):
        # s: [Batch, Seq_len, Channel] - seasonality
        # t: [Batch, Seq_len, Channel] - trend
        
        B, L, C = s.shape
        
        # --- Seasonal Stream (S-Mamba) ---
        # s: [B, L, C] -> Embedding -> [B, C, E]
        s_enc = self.s_embedding(s, None) 
        # [B, C, E] -> Encoder -> [B, C, E]
        s_enc, _ = self.s_encoder(s_enc, attn_mask=None)
        # [B, C, E] -> Linear -> [B, C, S] -> [B, S, C]
        s_out = self.s_projector(s_enc).permute(0, 2, 1)

        # --- Trend Stream (T-Mamba) ---
        # t: [B, L, C] -> Embedding -> [B, L, E]
        t_enc = self.t_embedding(t, None) # x_mark is None for now
        
        # [B, L, E] -> Encoder -> [B, L, E]
        t_enc, _ = self.t_encoder(t_enc, attn_mask=None)
        
        # Projector
        # [B, L, E] -> [B, E, L] -> Time Proj -> [B, E, S] -> [B, S, E]
        t_out = self.t_time_projector(t_enc.permute(0, 2, 1)).permute(0, 2, 1)
        # [B, S, E] -> Feat Proj -> [B, S, C]
        t_out = self.t_feature_projector(t_out)

        # --- Streams Concatenation ---
        # Align shapes for fusion
        # Both are [B, S, C]
        
        s_flat = s_out.permute(0, 2, 1).reshape(B * C, self.pred_len)
        t_flat = t_out.permute(0, 2, 1).reshape(B * C, self.pred_len)
        
        # Concatenate: [B*C, 2*S]
        x = torch.cat((s_flat, t_flat), dim=1)
        # Fusion: [B*C, S]
        x = self.fc8(x)
        
        # Final Reshape: [B, C, S] -> [B, S, C]
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)

        return x
