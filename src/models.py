from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class LSTMForecast(nn.Module):
    def __init__(self, input_dim: int, horizon: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.15) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class TransformerForecast(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pos(self.proj(x))
        z = self.encoder(z)
        pooled = torch.cat([z[:, -1], z.mean(dim=1)], dim=-1)
        return self.head(pooled)


class MultiScaleSeasonalResidualTransformer(nn.Module):
    """LSTM anchor plus zero-initialized multi-scale seasonal/weather residual correction."""

    def __init__(
        self,
        input_dim: int,
        horizon: int,
        weather_indices: list[int],
        target_index: int = 0,
        input_len: int = 90,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        dropout = 0.12 if horizon <= 90 else dropout
        self.horizon = horizon
        self.target_index = target_index
        self.weather_indices = weather_indices
        self.input_len = input_len
        self.use_dlinear_anchor = horizon > 90
        self.direct_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.direct_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )
        self.trend_linear = nn.Linear(input_len, horizon)
        self.seasonal_linear = nn.Linear(input_len, horizon)
        self.anchor_gate = nn.Sequential(
            nn.Linear(6, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon),
            nn.Sigmoid(),
        )
        self.proj = nn.Linear(input_dim, d_model)
        self.conv_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=k, padding=k // 2, groups=d_model),
                    nn.Conv1d(d_model, d_model, kernel_size=1),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for k in (3, 7, 15)
            ]
        )
        self.weather_gate = nn.Sequential(
            nn.Linear(len(weather_indices), d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool_norm = nn.LayerNorm(d_model * 3)
        self.residual_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, horizon),
        )
        self.stat_residual_head = nn.Sequential(
            nn.Linear(6, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )
        self.mix_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon),
            nn.Sigmoid(),
        )
        init_logit = -1.5 if horizon <= 90 else -3.0
        self.residual_logit = nn.Parameter(torch.tensor(init_logit))
        self._initialize_hybrid_heads()

    def _initialize_hybrid_heads(self) -> None:
        if self.use_dlinear_anchor:
            for head in (self.residual_head, self.stat_residual_head):
                final = head[-1]
                if isinstance(final, nn.Linear):
                    nn.init.zeros_(final.weight)
                    nn.init.zeros_(final.bias)
        gate_final = self.anchor_gate[-2]
        if isinstance(gate_final, nn.Linear):
            nn.init.zeros_(gate_final.weight)
            nn.init.constant_(gate_final.bias, 3.0)

    def _dlinear_anchor(self, target_seq: torch.Tensor) -> torch.Tensor:
        padded = F.pad(target_seq.unsqueeze(1), (12, 12), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=25, stride=1).squeeze(1)
        seasonal = target_seq - trend
        return self.trend_linear(trend) + self.seasonal_linear(seasonal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        direct_state, _ = self.direct_lstm(x)
        direct = self.direct_head(direct_state[:, -1])
        target_seq = x[:, :, self.target_index]

        z = self.proj(x)
        conv_in = z.transpose(1, 2)
        multiscale = torch.stack([branch(conv_in).transpose(1, 2) for branch in self.conv_branches], dim=0).mean(dim=0)

        weather_summary = x[:, :, self.weather_indices].mean(dim=1)
        gate = self.weather_gate(weather_summary).unsqueeze(1)
        z = z + multiscale * gate
        z = self.encoder(self.pos(z))

        last = z[:, -1]
        avg = z.mean(dim=1)
        attn_context = torch.softmax((z * last.unsqueeze(1)).sum(dim=-1) / math.sqrt(z.size(-1)), dim=1).unsqueeze(-1)
        context = (z * attn_context).sum(dim=1)
        pooled = self.pool_norm(torch.cat([last, avg, context], dim=-1))

        last_value = x[:, -1, self.target_index].unsqueeze(1).repeat(1, self.horizon)
        weekly = x[:, -7:, self.target_index]
        repeats = math.ceil(self.horizon / weekly.size(1))
        weekly_pattern = weekly.repeat(1, repeats)[:, : self.horizon]
        seasonal_mix = self.mix_head(pooled)
        baseline = seasonal_mix * weekly_pattern + (1.0 - seasonal_mix) * last_value

        ma7 = target_seq[:, -7:].mean(dim=1, keepdim=True)
        ma30 = target_seq[:, -30:].mean(dim=1, keepdim=True)
        ma90 = target_seq.mean(dim=1, keepdim=True)
        recent_slope = (target_seq[:, -1:] - target_seq[:, -30:-29]) / 29.0
        weekly_shift = target_seq[:, -7:-6] - target_seq[:, -1:]
        weather_summary = x[:, :, self.weather_indices].mean(dim=1)
        weather_level = weather_summary.mean(dim=1, keepdim=True)
        stat_features = torch.cat([ma7, ma30, ma90, recent_slope, weekly_shift, weather_level], dim=1)
        if self.use_dlinear_anchor:
            dlinear = self._dlinear_anchor(target_seq)
            anchor_gate = self.anchor_gate(stat_features)
            anchor = anchor_gate * direct + (1.0 - anchor_gate) * dlinear
            stat_residual = self.stat_residual_head(stat_features)
        else:
            anchor = direct
            stat_residual = 0.0

        seasonal_residual = baseline - last_value + self.residual_head(pooled) + stat_residual
        residual_scale = torch.sigmoid(self.residual_logit)
        return anchor + residual_scale * seasonal_residual


def build_model(name: str, input_dim: int, horizon: int, weather_indices: list[int]) -> nn.Module:
    name = name.lower()
    if name == "lstm":
        return LSTMForecast(input_dim=input_dim, horizon=horizon)
    if name == "transformer":
        return TransformerForecast(input_dim=input_dim, horizon=horizon)
    if name in {"msrt", "custom", "ours"}:
        return MultiScaleSeasonalResidualTransformer(
            input_dim=input_dim,
            horizon=horizon,
            weather_indices=weather_indices,
        )
    raise ValueError(f"Unknown model: {name}")
