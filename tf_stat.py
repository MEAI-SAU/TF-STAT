import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBranch(nn.Module):
    """时域多尺度卷积：3/5/7 并联，三路拼接成 d_model。"""
    def __init__(self, d_model, kernel_sizes=(3, 5, 7)):
        super().__init__()
        d = d_model // 3
        d_last = d_model - 2 * d
        k1, k2, k3 = kernel_sizes
        self.conv3 = nn.Sequential(nn.Conv1d(1, d, k1, padding=k1 // 2, bias=False),
                                   nn.BatchNorm1d(d), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv1d(1, d, k2, padding=k2 // 2, bias=False),
                                   nn.BatchNorm1d(d), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv1d(1, d_last, k3, padding=k3 // 2, bias=False),
                                   nn.BatchNorm1d(d_last), nn.ReLU(inplace=True))

    def forward(self, x):                       # (B, 12, T)
        B, L, T = x.shape
        x = x.reshape(B * L, 1, T)
        out = torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)
        return out.permute(0, 2, 1).reshape(B, L, T, -1)

class FrequencyBranch(nn.Module):
    """STFT → 2D CNN → 频率轴池化 → 线性投影回 d_model。"""
    def __init__(self, d_model, n_fft=256, hop_length=64, target_T=1000):
        super().__init__()
        self.n_fft, self.hop_length, self.target_T = n_fft, hop_length, target_T
        self.cnn2d = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.proj = nn.Linear(64, d_model)

    def forward(self, x):
        B, L, T = x.shape
        window = torch.hann_window(self.n_fft, device=x.device)
        spec = torch.stft(x.reshape(B * L, T).float(), n_fft=self.n_fft,
                          hop_length=self.hop_length, win_length=self.n_fft,
                          window=window, return_complex=True).abs().unsqueeze(1)
        ff = self.freq_pool(self.cnn2d(spec)).squeeze(2)
        if ff.shape[-1] != self.target_T:
            ff = F.interpolate(ff, size=self.target_T, mode="linear", align_corners=False)
        return self.proj(ff.permute(0, 2, 1)).reshape(B, L, T, -1)

class GatedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, d_model)

    def forward(self, ft, ff):
        a = torch.sigmoid(self.gate(torch.cat([ft, ff], dim=-1)))
        return a * ft + (1 - a) * ff

class SpatialAttention(nn.Module):
    """跨 12 导联的注意力（每个时间步独立）。"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads, self.d_h = num_heads, d_model // num_heads
        self.scale = math.sqrt(self.d_h)
        self.wq, self.wk, self.wv = (nn.Linear(d_model, d_model) for _ in range(3))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, T, D = x.shape
        xr = x.permute(0, 2, 1, 3).reshape(B * T, L, D)
        head = lambda w: w(xr).reshape(B * T, L, self.num_heads, self.d_h).permute(0, 2, 1, 3)
        q, k, v = head(self.wq), head(self.wk), head(self.wv)
        attn = F.softmax(q @ k.transpose(-1, -2) / self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * T, L, D)
        return self.out_proj(out).reshape(B, T, L, D).permute(0, 2, 1, 3)

class CascadedTemporalHead(nn.Module):
    """级联时间注意力的单个头：q 在上一头输出上再细化一次。"""
    def __init__(self, d_h):
        super().__init__()
        self.scale = math.sqrt(d_h)
        self.q_proj = nn.Conv1d(d_h, d_h, 1)
        self.k_proj = nn.Conv1d(d_h, d_h, 1)
        self.v_proj = nn.Conv1d(d_h, d_h, 1)
        self.q_refine = nn.Conv1d(d_h, d_h, 1)

    def forward(self, z):                       # (N, d_h, T)
        q = self.q_refine(self.q_proj(z)).transpose(1, 2)
        k = self.k_proj(z).transpose(1, 2)
        v = self.v_proj(z).transpose(1, 2)
        return torch.bmm(F.softmax(q @ k.transpose(1, 2) / self.scale, dim=-1), v)

class CascadedTemporalAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_h, self.num_heads = d_model // num_heads, num_heads
        self.heads = nn.ModuleList([CascadedTemporalHead(self.d_h) for _ in range(num_heads)])
        self.out_proj = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x):
        prev, outs = None, []
        for i, (head, g) in enumerate(zip(self.heads, x.split(self.d_h, dim=-1))):
            z = g if i == 0 else g + prev          # 级联：上一头的输出传给下一头
            prev = head(z.transpose(1, 2))
            outs.append(prev)
        y = torch.cat(outs, dim=-1).transpose(1, 2)
        return self.out_proj(F.relu(y)).transpose(1, 2)

class DecoupledBlock(nn.Module):
    """空间与时间注意力解耦并联，再各自残差 + FFN。"""
    def __init__(self, d_model, num_heads, ffn_dim, dropout=0.3):
        super().__init__()
        self.spatial = SpatialAttention(d_model, num_heads)
        self.temporal = CascadedTemporalAttention(d_model, num_heads)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, d_model), nn.Dropout(dropout))

    def forward(self, x):
        B, L, T, D = x.shape
        Hs = self.spatial(x)
        Ht = self.temporal(x.reshape(B * L, T, D)).reshape(B, L, T, D)
        h = self.norm1(x + Hs + Ht)
        return self.norm2(h + self.ffn(h))

class TFSTAT(nn.Module):
    def __init__(self, num_classes=2, d_model=512, num_heads=8, num_layers=1,
                 ffn_dim=2048, dropout=0.3, num_leads=12, seq_len=1000,
                 n_fft=256, hop_length=64, kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.tf_repr = nn.ModuleDict({
            "temporal": TemporalBranch(d_model, kernel_sizes),
            "freq": FrequencyBranch(d_model, n_fft, hop_length, seq_len),
            "fusion": GatedFusion(d_model)})
        self.blocks = nn.ModuleList([DecoupledBlock(d_model, num_heads, ffn_dim, dropout)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.ReLU(inplace=True),
                                        nn.Dropout(dropout), nn.Linear(d_model // 2, num_classes))

    def forward(self, x):
        f = self.tf_repr["fusion"](self.tf_repr["temporal"](x), self.tf_repr["freq"](x))
        for blk in self.blocks:
            f = blk(f)
        return self.classifier(self.dropout(self.norm(f).mean(dim=(1, 2))))