import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ===== Temporal branch: multiscale convolutions extract local features from each lead =====
class TemporalBranch(nn.Module):
    def __init__(self, d_model: int, kernel_sizes: tuple[int, ...]) -> None:
        super().__init__()
        if not kernel_sizes or d_model < len(kernel_sizes):
            raise ValueError("Embedding dimension must provide channels for every convolution branch")
        if any(kernel < 1 or kernel % 2 == 0 for kernel in kernel_sizes):
            raise ValueError("Convolution kernel sizes must be positive odd integers")

        widths = [d_model // len(kernel_sizes)] * len(kernel_sizes)
        widths[-1] += d_model - sum(widths)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(1, width, kernel, padding=kernel // 2, bias=False),
                nn.BatchNorm1d(width),
                nn.ReLU(inplace=True),
            )
            for width, kernel in zip(widths, kernel_sizes)
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, leads, samples = x.shape
        signal = x.reshape(batch * leads, 1, samples)
        features = torch.cat([branch(signal) for branch in self.branches], dim=1)
        return features.transpose(1, 2).reshape(batch, leads, samples, -1)


# ===== Frequency branch: short-time Fourier transform and 2D convolutions =====
class FrequencyBranch(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_fft: int,
        hop_length: int,
        cnn_channels: tuple[int, int],
    ) -> None:
        super().__init__()
        if n_fft < 2 or hop_length < 1 or len(cnn_channels) != 2 or any(
            width < 1 for width in cnn_channels
        ):
            raise ValueError("Invalid frequency branch configuration")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(cnn_channels[0], cnn_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(cnn_channels[1], d_model)

    def forward(self, x: Tensor) -> Tensor:
        batch, leads, samples = x.shape
        spectrum = torch.stft(
            x.reshape(batch * leads, samples).float(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.float(),
            return_complex=True,
        ).abs().unsqueeze(1)
        features = self.cnn(spectrum.to(self.cnn[0].weight.dtype)).mean(dim=2)
        features = F.interpolate(features, size=samples, mode="linear", align_corners=False)
        return self.proj(features.transpose(1, 2)).reshape(batch, leads, samples, -1)


# ===== Gated fusion: adaptively combine temporal and frequency features =====
class GatedFusion(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, temporal: Tensor, frequency: Tensor) -> Tensor:
        weight = torch.sigmoid(self.gate(torch.cat((temporal, frequency), dim=-1)))
        return weight * temporal + (1 - weight) * frequency


# ===== Spatial attention: exchange information across leads at each time step =====
class SpatialAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, return_lead_scores: bool = False) -> tuple[Tensor, Tensor | None]:
        batch, leads, samples, width = x.shape
        tokens = x.permute(0, 2, 1, 3).reshape(batch * samples, leads, width)

        def split(projection: nn.Linear) -> Tensor:
            return projection(tokens).reshape(
                batch * samples, leads, self.num_heads, self.head_dim
            ).transpose(1, 2)

        query, key, value = split(self.query), split(self.key), split(self.value)
        lead_scores = None
        if return_lead_scores:
            weights = torch.softmax(
                query @ key.transpose(-1, -2) / math.sqrt(self.head_dim), dim=-1
            )
            attended = weights @ value
            lead_scores = weights.mean(dim=(1, 2)).reshape(batch, samples, leads).mean(dim=1)
        else:
            attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)

        attended = attended.transpose(1, 2).reshape(batch * samples, leads, width)
        result = self.output(attended).reshape(batch, samples, leads, width)
        return result.permute(0, 2, 1, 3), lead_scores


# ===== Single cascaded temporal head: generate Q/K/V and refine the query =====
class CascadedTemporalHead(nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.query = nn.Conv1d(head_dim, head_dim, 1)
        self.key = nn.Conv1d(head_dim, head_dim, 1)
        self.value = nn.Conv1d(head_dim, head_dim, 1)
        self.query_refine = nn.Conv1d(head_dim, head_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        channels_first = x.transpose(1, 2)
        query = self.query_refine(self.query(channels_first)).transpose(1, 2)
        key = self.key(channels_first).transpose(1, 2)
        value = self.value(channels_first).transpose(1, 2)
        return F.scaled_dot_product_attention(
            query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1), dropout_p=0.0
        ).squeeze(1)


# ===== Cascaded temporal attention: each head receives the previous head's output =====
class CascadedTemporalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList(CascadedTemporalHead(head_dim) for _ in range(num_heads))
        self.output = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        previous = None
        outputs = []
        for group, head in zip(x.chunk(len(self.heads), dim=-1), self.heads):
            previous = head(group if previous is None else group + previous)
            outputs.append(previous)
        joined = torch.cat(outputs, dim=-1)
        return self.output(F.relu(joined).transpose(1, 2)).transpose(1, 2)


# ===== Decoupled block: model inter-lead and temporal relationships in parallel =====
class DecoupledBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.spatial = SpatialAttention(d_model, num_heads)
        self.temporal = CascadedTemporalAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, return_lead_scores: bool = False) -> tuple[Tensor, Tensor | None]:
        batch, leads, samples, width = x.shape
        spatial, lead_scores = self.spatial(x, return_lead_scores)
        temporal = self.temporal(x.reshape(batch * leads, samples, width)).reshape(
            batch, leads, samples, width
        )
        hidden = self.norm1(x + spatial + temporal)
        return self.norm2(hidden + self.ffn(hidden)), lead_scores


# ===== Core network: pass fused features through decoupled blocks sequentially =====
class TFSTAT(nn.Module):
    def __init__(
        self,
        *,
        num_leads: int,
        seq_len: int,
        num_classes: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        classifier_dim: int,
        kernel_sizes: tuple[int, ...],
        n_fft: int,
        hop_length: int,
        cnn_channels: tuple[int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model < 1 or num_heads < 1 or d_model % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by the number of attention heads")
        if num_layers < 1 or seq_len < n_fft or num_leads < 1:
            raise ValueError("Invalid number of layers, input length, or number of leads")
        if num_classes < 1 or classifier_dim < 1 or ffn_dim < 1:
            raise ValueError("Invalid classifier or feed-forward network dimension")

        self.num_leads = num_leads
        self.seq_len = seq_len
        self.temporal = TemporalBranch(d_model, kernel_sizes)
        self.frequency = FrequencyBranch(d_model, n_fft, hop_length, cnn_channels)
        self.fusion = GatedFusion(d_model)
        self.blocks = nn.ModuleList(
            DecoupledBlock(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, classifier_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes),
        )

    def forward(self, x: Tensor, return_lead_scores: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        if x.ndim != 3 or x.shape[1:] != (self.num_leads, self.seq_len):
            raise ValueError("Input shape does not match the model configuration")

        features = self.fusion(self.temporal(x), self.frequency(x))
        scores = []
        for block in self.blocks:
            features, lead_scores = block(features, return_lead_scores)
            if lead_scores is not None:
                scores.append(lead_scores)

        logits = self.classifier(self.dropout(self.norm(features).mean(dim=(1, 2))))
        if return_lead_scores:
            return logits, torch.stack(scores).mean(dim=0)
        return logits
