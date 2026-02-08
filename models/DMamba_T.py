import torch
import torch.nn as nn
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding
from mamba_ssm import Mamba

class Model(nn.Module):
    """
    Time-series Mamba: Models temporal dependencies directly.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = getattr(configs, 'output_attention', False)
        self.use_norm = getattr(configs, 'use_norm', False)

        # Embedding: [B, L, N] -> [B, L, E]
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        self.class_strategy = getattr(configs, 'class_strategy', None)

        # Encoder-only architecture
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
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # Projector
        # 1) Time dimension: seq_len -> pred_len
        self.time_projector = nn.Linear(configs.seq_len, configs.pred_len, bias=True)
        # 2) Feature dimension: d_model -> enc_in (N)
        self.feature_projector = nn.Linear(configs.d_model, configs.enc_in, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Embedding: B L N -> B L E
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encoder: B L E -> B L E
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # Projector
        # Step 1: Time dimension L -> S
        dec_out = self.time_projector(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        # Step 2: Feature dimension E -> N
        dec_out = self.feature_projector(dec_out)

        if self.use_norm:
            # De-Normalization
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
