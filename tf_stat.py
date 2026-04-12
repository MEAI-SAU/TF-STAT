import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBranch(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        d = d_model // 3
        d_last = d_model - 2 * d
        self.conv3 = nn.Sequential(
            nn.Conv1d(1, d, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(1, d, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(1, d_last, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(d_last),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, T = x.shape
        x = x.reshape(B * L, 1, T)
        out = torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)
        return out.permute(0, 2, 1).reshape(B, L, T, -1)


class FrequencyBranch(nn.Module):
    def __init__(self, d_model: int, n_fft: int = 256, hop_length: int = 64, target_T: int = 1000):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_T = target_T
        self.cnn2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.proj = nn.Linear(64, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, T = x.shape
        x_flat = x.reshape(B * L, T).float()
        window = torch.hann_window(self.n_fft, device=x.device)
        specs = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
        ).abs()
        specs = specs.unsqueeze(1)
        ff = self.cnn2d(specs)
        ff = self.freq_pool(ff).squeeze(2)
        if ff.shape[-1] != self.target_T:
            ff = F.interpolate(ff, size=self.target_T, mode='linear', align_corners=False)
        ff = ff.permute(0, 2, 1)
        return self.proj(ff).reshape(B, L, T, -1)


class GatedFusion(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, d_model)

    def forward(self, ft: torch.Tensor, ff: torch.Tensor) -> torch.Tensor:
        alpha = torch.sigmoid(self.gate(torch.cat([ft, ff], dim=-1)))
        return alpha * ft + (1.0 - alpha) * ff


class TimeFrequencyJointRepresentation(nn.Module):
    def __init__(self, d_model: int, n_fft: int = 256, hop_length: int = 64, target_T: int = 1000):
        super().__init__()
        self.temporal = TemporalBranch(d_model)
        self.freq = FrequencyBranch(d_model, n_fft, hop_length, target_T)
        self.fusion = GatedFusion(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion(self.temporal(x), self.freq(x))


class SpatialAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_h = d_model // num_heads
        self.scale = math.sqrt(self.d_h)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        B, L, T, D = x.shape
        x_r = x.permute(0, 2, 1, 3).reshape(B * T, L, D)

        def proj_heads(w, inp):
            return w(inp).reshape(B * T, L, self.num_heads, self.d_h).permute(0, 2, 1, 3)

        Q = proj_heads(self.wq, x_r)
        K = proj_heads(self.wk, x_r)
        V = proj_heads(self.wv, x_r)
        attn = F.softmax(torch.matmul(Q, K.transpose(-1, -2)) / self.scale, dim=-1)
        out = torch.matmul(attn, V).permute(0, 2, 1, 3).reshape(B * T, L, D)
        out = self.out_proj(out).reshape(B, T, L, D).permute(0, 2, 1, 3)
        return out, attn.reshape(B, T, self.num_heads, L, L)


class CascadedTemporalAttentionHead(nn.Module):
    def __init__(self, d_h: int):
        super().__init__()
        self.scale = math.sqrt(d_h)
        self.q_proj = nn.Conv1d(d_h, d_h, kernel_size=1)
        self.k_proj = nn.Conv1d(d_h, d_h, kernel_size=1)
        self.v_proj = nn.Conv1d(d_h, d_h, kernel_size=1)
        self.q_refine = nn.Conv1d(d_h, d_h, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        q = self.q_refine(self.q_proj(z)).transpose(1, 2)
        k = self.k_proj(z).transpose(1, 2)
        v = self.v_proj(z).transpose(1, 2)
        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) / self.scale, dim=-1)
        return torch.bmm(attn, v)


class CascadedTemporalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_h = d_model // num_heads
        self.num_heads = num_heads
        self.heads = nn.ModuleList(
            [CascadedTemporalAttentionHead(self.d_h) for _ in range(num_heads)]
        )
        self.out_proj = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        groups = x.split(self.d_h, dim=-1)
        prev = None
        outputs = []
        for i, (head, g) in enumerate(zip(self.heads, groups)):
            z = g if i == 0 else g + prev
            out = head(z.transpose(1, 2))
            prev = out
            outputs.append(out)
        y = torch.cat(outputs, dim=-1).transpose(1, 2)
        return self.out_proj(F.relu(y)).transpose(1, 2)


class SpatiotemporalDecoupledBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
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

    def forward(self, x: torch.Tensor):
        B, L, T, D = x.shape
        Hs, weights = self.spatial(x)
        Ht = self.temporal(x.reshape(B * L, T, D)).reshape(B, L, T, D)
        h = self.norm1(x + Hs + Ht)
        return self.norm2(h + self.ffn(h)), weights


class ECGAugmentation(nn.Module):
    def __init__(self, shift: int = 50, noise_std: float = 0.05, mask_prob: float = 0.1):
        super().__init__()
        self.shift = shift
        self.noise_std = noise_std
        self.mask_prob = mask_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B, L, T = x.shape
        shifts = torch.randint(-self.shift, self.shift + 1, (B,))
        x = torch.stack([torch.roll(x[i], shifts[i].item(), dims=-1) for i in range(B)])
        x = x + torch.randn_like(x) * self.noise_std
        mask = (torch.rand(B, L, 1, device=x.device) > self.mask_prob).float()
        return x * mask


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum()


class TFSTAT(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        ffn_dim: int = 2048,
        dropout: float = 0.3,
        num_leads: int = 12,
        seq_len: int = 1000,
        n_fft: int = 256,
        hop_length: int = 64,
    ):
        super().__init__()
        self.augment = ECGAugmentation()
        self.tf_repr = TimeFrequencyJointRepresentation(d_model, n_fft, hop_length, seq_len)
        self.blocks = nn.ModuleList(
            [SpatiotemporalDecoupledBlock(d_model, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        self._spatial_weights: list = []
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.augment(x)
        self._spatial_weights = []
        feat = self.tf_repr(x)
        for block in self.blocks:
            feat, w = block(feat)
            self._spatial_weights.append(w)
        feat = self.norm(feat)
        out = feat.mean(dim=(1, 2))
        return self.classifier(self.dropout(out))

    @torch.no_grad()
    def get_lead_contributions(self) -> torch.Tensor | None:
        if not self._spatial_weights:
            return None
        all_w = torch.stack(self._spatial_weights, dim=0).mean(dim=0)
        contrib = all_w.mean(dim=(1, 2, 3))
        return contrib / contrib.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def compute_alignment(
        self, contributions: torch.Tensor, mi_type: str = 'anterior', top_k: int = 4
    ) -> float:
        criteria = {
            'anterior': {6, 7, 8, 9},
            'inferior': {1, 2, 5},
            'lateral': {0, 4, 10, 11},
        }
        standard = criteria.get(mi_type, set())
        if not standard:
            return 0.0
        top_indices = contributions.topk(top_k, dim=-1).indices
        scores = []
        for b in range(top_indices.shape[0]):
            pred = set(top_indices[b].tolist())
            scores.append(len(pred & standard) / len(standard))
        return sum(scores) / len(scores)


def build_tfstat(num_classes: int = 2, **kwargs) -> TFSTAT:
    defaults = dict(
        d_model=512,
        num_heads=8,
        num_layers=4,
        ffn_dim=2048,
        dropout=0.3,
        num_leads=12,
        seq_len=1000,
        n_fft=256,
        hop_length=64,
    )
    defaults.update(kwargs)
    return TFSTAT(num_classes=num_classes, **defaults)


def build_optimizer(model: TFSTAT, lr: float = 3e-4, weight_decay: float = 0.01):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, T_max: int = 50, eta_min: float = 1e-6):
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_max // 5, T_mult=1, eta_min=eta_min
    )


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = 'max'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best = None
        self.stopped = False

    def step(self, metric: float, model: nn.Module, path: str = 'best_model.pt') -> bool:
        improved = (
            self.best is None
            or (self.mode == 'max' and metric > self.best)
            or (self.mode == 'min' and metric < self.best)
        )
        if improved:
            self.best = metric
            self.counter = 0
            torch.save(model.state_dict(), path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
        return self.stopped


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    TEST_SEQ_LEN = 200
    TEST_D_MODEL = 128
    TEST_NUM_HEADS = 4
    TEST_NUM_LAYERS = 2
    TEST_FFN_DIM = 256
    TEST_BATCH = 2

    model = build_tfstat(
        num_classes=2,
        d_model=TEST_D_MODEL,
        num_heads=TEST_NUM_HEADS,
        num_layers=TEST_NUM_LAYERS,
        ffn_dim=TEST_FFN_DIM,
        seq_len=TEST_SEQ_LEN,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters (test config)    : {total:,}')
    print(f'Trainable parameters (test config): {trainable:,}')

    full_model = build_tfstat(num_classes=2)
    full_total = sum(p.numel() for p in full_model.parameters())
    print(f'Total parameters (full config d=512, L=4): {full_total:,}')
    del full_model

    model.train()
    x = torch.randn(TEST_BATCH, 12, TEST_SEQ_LEN, device=device)
    labels = torch.randint(0, 2, (TEST_BATCH,), device=device)
    logits = model(x)
    print(f'Logits shape: {logits.shape}')

    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    loss = loss_fn(logits, labels)
    loss.backward()
    print(f'Loss: {loss.item():.4f}')
    print('Backward pass OK')

    model.eval()
    with torch.no_grad():
        logits = model(x)
        contrib = model.get_lead_contributions()
        if contrib is not None:
            print(f'Lead contributions (sample 0): {contrib[0].cpu().numpy().round(3)}')
            align = model.compute_alignment(contrib, mi_type='anterior')
            print(f'Anterior MI alignment: {align:.3f}')

    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    early_stop = EarlyStopping(patience=10, mode='max')
    print('Optimizer, scheduler, and early stopping initialized successfully.')
    print('All checks passed.')
