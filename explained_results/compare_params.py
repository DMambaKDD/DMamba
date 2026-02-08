import torch
import torch.nn as nn
from layers.network_mlp import NetworkMLP
from layers.network_mamba import Network_Mamba
from mamba_ssm import Mamba
from layers.Embed import DataEmbedding_inverted
from layers.Mamba_EncDec import Encoder, EncoderLayer

class Config:
    def __init__(self, seq_len=96, pred_len=96, enc_in=7, d_model=512, d_state=16, d_ff=2048, e_layers=2, dropout=0.05, activation='gelu', embed='timeF', freq='h'):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.d_state = d_state
        self.d_ff = d_ff
        self.e_layers = e_layers
        self.dropout = dropout
        self.activation = activation
        self.embed = embed
        self.freq = freq

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compare_seasonal_params():
    configs = Config()
    
    # 1. MLP Seasonal Channel (from DMamba_MLP)
    # DMamba_MLP uses NetworkMLP for both seasonal and trend. 
    # So the "seasonal channel" is just one instance of NetworkMLP.
    mlp_seasonal = NetworkMLP(configs.seq_len, configs.pred_len)
    mlp_params = count_parameters(mlp_seasonal)
    
    # 2. DMamba Seasonal Channel (from DMamba)
    # In Network_Mamba, the seasonal part is the "Non-linear Stream (Mamba logic)"
    # It consists of: enc_embedding, encoder (Mamba layers), projector
    # We need to extract just these parts to count parameters fairly.
    
    # Re-instantiate the parts corresponding to DMamba seasonal stream
    dmamba_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
    dmamba_encoder = Encoder(
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
    dmamba_projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
    
    dmamba_params = count_parameters(dmamba_embedding) + count_parameters(dmamba_encoder) + count_parameters(dmamba_projector)

    print(f"--- Parameter Count Comparison (Seasonal Channel Only) ---")
    print(f"Configuration: seq_len={configs.seq_len}, pred_len={configs.pred_len}, d_model={configs.d_model}, e_layers={configs.e_layers}")
    print(f"DMamba_MLP (Seasonal MLP): {mlp_params:,} parameters")
    print(f"DMamba (Seasonal Mamba): {dmamba_params:,} parameters")
    print(f"Ratio (Mamba / MLP): {dmamba_params / mlp_params:.2f}x")
    
    # Detailed breakdown for DMamba
    print(f"\n--- DMamba Breakdown ---")
    print(f"Embedding: {count_parameters(dmamba_embedding):,}")
    print(f"Encoder (Mamba Layers): {count_parameters(dmamba_encoder):,}")
    print(f"Projector: {count_parameters(dmamba_projector):,}")

if __name__ == "__main__":
    compare_seasonal_params()
