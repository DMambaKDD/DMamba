import torch
from torch import nn
from mamba_ssm import Mamba
from layers.Embed import DataEmbedding_inverted
from layers.Mamba_EncDec import Encoder, EncoderLayer

class Network_Mamba(nn.Module):
    def __init__(self, configs):
        super(Network_Mamba, self).__init__()

        # Parameters
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in

        # --- Non-linear Stream (Mamba logic) ---
        # Embedding: Projects the time dimension of each variate into d_model
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        # Encoder: Mamba layers to model inter-variate dependencies
        self.encoder = Encoder(
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
        
        # Project back to pred_len
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # --- Linear Stream (Trend Stream) ---
        # MLP based channel-independent processing
        self.fc5 = nn.Linear(configs.seq_len, configs.pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(configs.pred_len * 2)

        self.fc6 = nn.Linear(configs.pred_len * 2, configs.pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(configs.pred_len // 2)

        self.fc7 = nn.Linear(configs.pred_len // 2, configs.pred_len)

        # --- Streams Concatination ---
        self.fc8 = nn.Linear(configs.pred_len * 2, configs.pred_len)

    def forward(self, s, t):
        # s: [Batch, Seq_len, Channel] - seasonality
        # t: [Batch, Seq_len, Channel] - trend
        
        B, L, C = s.shape
        
        # 1. Non-linear Stream (Mamba)
        # s: [B, L, C] -> Embedding -> [B, C, E]
        s_enc = self.enc_embedding(s, None) 
        # [B, C, E] -> Encoder -> [B, C, E]
        s_enc, _ = self.encoder(s_enc, attn_mask=None)
        # [B, C, E] -> Linear -> [B, C, S] -> [B, S, C]
        s_out = self.projector(s_enc).permute(0, 2, 1)

        # 2. Linear Stream (Trend - Channel Independent)
        # t: [B, L, C] -> [B, C, L] -> [B*C, L]
        t = t.permute(0, 2, 1).reshape(B * C, L)
        
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t) # [B*C, S]
        
        # Reshape back to [B, S, C]
        t_out = t.reshape(B, C, self.pred_len).permute(0, 2, 1)

        # 3. Streams Concatination
        s_flat = s_out.permute(0, 2, 1).reshape(B * C, self.pred_len)
        t_flat = t_out.permute(0, 2, 1).reshape(B * C, self.pred_len)
        
        x = torch.cat((s_flat, t_flat), dim=1)
        x = self.fc8(x)
        
        # Final Reshape
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)

        return x
