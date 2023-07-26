import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# B: Batch size
# L: Lookback
# H: Horizon
# N: the number of series
# r: the number of covariates for each series
# r_hat: temporalWidth in the paper, i.e., \hat{r} << r
# p: decoderOutputDim in the paper
# hidden_dim: hiddenSize in the paper


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)
        self.linear_res = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: [B, L, in_dim] or [B, in_dim]
        h = F.relu(self.linear_1(x))
        h = self.dropout(self.linear_2(h))
        res = self.linear_res(x)
        out = self.layernorm(h + res)
        return out


class Encoder(nn.Module):
    def __init__(self, layer_num, hidden_dim, r, r_hat, L, H, featureProjectionHidden):
        super(Encoder, self).__init__()
        self.encoder_layer_num = layer_num
        self.horizon = H
        self.feature_projection = ResidualBlock(r, featureProjectionHidden, r_hat)
        self.first_encoder_layer = ResidualBlock(L+1+(L+H)*r_hat, hidden_dim, hidden_dim)
        self.other_encoder_layers = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, hidden_dim) for _ in range(layer_num-1)
        ])

    def forward(self, x, covariates, attributes):
        # x: [B*N,L], covariates: [B*N,1], attributes: [B*N,L+H,r]

        # Feature Projection
        covariates = self.feature_projection(covariates)
        covariates_future = covariates[:, -self.horizon, :]

        # Flatten
        covariates_flat = rearrange(covariates, 'b l r -> b (l r)')

        # Concat
        e = torch.cat([x, attributes, covariates_flat], dim=1)

        # Dense Encoder
        e = self.first_encoder_layer(e)
        for i in range(self.encoder_layer_num-1):
            e = self.other_encoder_layers[i](e)

        # e: [B*N,hidden_dim], covariates_future: [B*N,H,r_hat]
        return e, covariates_future


class Decoder(nn.Module):
    def __init__(self, layer_num, hidden_dim, r_hat, H, p, temporalDecoderHidden):
        super(Decoder, self).__init__()
        self.decoder_layer_num = layer_num
        self.horizon = H
        self.last_decoder_layer = ResidualBlock(hidden_dim, hidden_dim, p * H)
        self.other_decoder_layers = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim, hidden_dim) for _ in range(layer_num-1)
        ])
        self.temporaldecoer = ResidualBlock(p + r_hat, temporalDecoderHidden, 1)

    def forward(self, e, covariates_future):
        for i in range(self.decoder_layer_num-1):
            e = self.other_decoder_layers[i](e)
        g = self.last_decoder_layer(e)

        # Unflatten
        matrixD = rearrange(g, 'b (h p) -> b h p', h=self.horizon)

        # Stack
        out = torch.cat([matrixD, covariates_future], dim=-1)

        # Temporal Decoder
        out = self.temporaldecoer(out)

        return out


