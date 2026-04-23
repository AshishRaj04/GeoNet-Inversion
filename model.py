import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DROP PATH (STOCHASTIC DEPTH)
# ============================================================

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Stochastic depth per sample during training."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor + keep_prob)
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return f"drop_prob={self.drop_prob:.3f}"


# ============================================================
# CNN COMPONENTS
# ============================================================

class ConvBlock(nn.Module):
    """Double convolution block: Conv → BN → GELU → Conv → BN → GELU.
    
    The workhorse of the encoder and decoder. Uses GELU (smoother than ReLU)
    for better gradient flow in hybrid architectures.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ResConvBlock(nn.Module):
    """Residual double convolution block.
    
    Adds a skip connection around the ConvBlock for better gradient flow
    in deeper networks. The 1×1 conv handles channel mismatch.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


# ============================================================
# SWIN TRANSFORMER COMPONENTS (for bottleneck only)
# ============================================================

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """Partition (B, H, W, C) into windows of (wh, ww)."""
    B, H, W, C = x.shape
    wh, ww = window_size
    x = x.view(B, H // wh, wh, W // ww, ww, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, wh, ww, C)


def window_reverse(windows, window_size, H, W):
    """Reverse window partition back to (B, H, W, C)."""
    wh, ww = window_size
    B = int(windows.shape[0] / (H * W / wh / ww))
    x = windows.view(B, H // wh, W // ww, wh, ww, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class SwinBlock(nn.Module):
    """Single Swin Transformer block with optional window shifting and DropPath."""
    def __init__(self, dim, num_heads, window_size=(7, 7), shift_size=(0, 0),
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path_rate=0.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size, num_heads, attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Pad to window-divisible size
        wh, ww = self.window_size
        pad_b = (wh - H % wh) % wh
        pad_r = (ww - W % ww) % ww
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = x.shape[1], x.shape[2]

        # Shift
        shift_h, shift_w = self.shift_size
        if shift_h > 0 or shift_w > 0:
            shifted_x = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(1, 2))
            attn_mask = self._compute_mask(Hp, Wp, x.device)
        else:
            shifted_x = x
            attn_mask = None

        # Window attention
        x_win = window_partition(shifted_x, self.window_size).view(-1, wh * ww, C)
        attn_win = self.attn(x_win, mask=attn_mask).view(-1, wh, ww, C)
        shifted_x = window_reverse(attn_win, self.window_size, Hp, Wp)

        # Reverse shift
        if shift_h > 0 or shift_w > 0:
            x = torch.roll(shifted_x, shifts=(shift_h, shift_w), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding
        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def _compute_mask(self, Hp, Wp, device):
        wh, ww = self.window_size
        sh, sw = self.shift_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        for cnt, (h_slice, w_slice) in enumerate([
            (slice(0, -wh), slice(0, -ww)), (slice(0, -wh), slice(-ww, -sw)),
            (slice(0, -wh), slice(-sw, None)), (slice(-wh, -sh), slice(0, -ww)),
            (slice(-wh, -sh), slice(-ww, -sw)), (slice(-wh, -sh), slice(-sw, None)),
            (slice(-sh, None), slice(0, -ww)), (slice(-sh, None), slice(-ww, -sw)),
            (slice(-sh, None), slice(-sw, None)),
        ]):
            img_mask[:, h_slice, w_slice, :] = cnt

        mask_win = window_partition(img_mask, self.window_size).view(-1, wh * ww)
        attn_mask = mask_win.unsqueeze(1) - mask_win.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)


class SwinBottleneck(nn.Module):
    """Swin Transformer bottleneck: processes CNN features with global attention.
    
    Takes CNN feature maps (B, C, H, W), reshapes to tokens (B, H*W, C),
    applies multiple Swin blocks for global reasoning, then reshapes back.
    
    This is where the Transformer adds value — understanding relationships
    between distant geological features.
    """
    def __init__(self, dim, depth=6, num_heads=8, window_size=(5, 4),
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path_rate=0.1):
        super().__init__()

        # Linear DropPath schedule across blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = (0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2)
            self.blocks.append(
                SwinBlock(
                    dim=dim, num_heads=num_heads, window_size=window_size,
                    shift_size=shift, mlp_ratio=mlp_ratio,
                    drop=drop, attn_drop=attn_drop, drop_path_rate=dpr[i],
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) — CNN feature maps
        Returns:
            x: (B, C, H, W) — globally-enhanced feature maps
        """
        B, C, H, W = x.shape

        # CNN format → Transformer format
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        for blk in self.blocks:
            x = blk(x, H, W)

        x = self.norm(x)

        # Transformer format → CNN format
        x = x.transpose(1, 2).view(B, C, H, W)
        return x


# ============================================================
# HYBRID CNN-SWIN UNET
# ============================================================

class HybridSwinUNet(nn.Module):
    """Hybrid CNN-Swin UNet V3.5.
    
    Architecture philosophy:
    - CNN Encoder: Extracts LOCAL features (wave patterns, edges, faults)
      CNNs have strong inductive bias for local, translation-invariant patterns
      which perfectly matches seismic wave propagation physics.
      
    - Swin Bottleneck: Captures GLOBAL relationships (layer correlations,
      long-range fault connections, basin-wide velocity trends)
      Transformers excel at relating distant spatial locations.
      
    - CNN Decoder: Reconstructs SHARP outputs (precise boundaries, crisp faults)
      CNNs naturally produce spatially coherent, sharp outputs unlike
      transformers which tend to over-smooth.
    
    Input:  (B, 5, 1000, 70)  — 5 seismic sources, 1000 timesteps, 70 receivers
    Output: (B, 70, 70)       — 2D velocity model
    """
    def __init__(
        self,
        in_ch=5,
        out_ch=1,
        base_ch=64,
        bottleneck_depth=6,
        bottleneck_heads=8,
        window_size=(5, 4),
        mlp_ratio=4.0,
        drop_path_rate=0.1,
    ):
        super().__init__()

        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 6
        # c1=64, c2=128, c3=256, c4=384

        # ── CNN ENCODER ──
        # Each stage: ResConvBlock (local features) + MaxPool (downsample)
        self.enc1 = ResConvBlock(in_ch, c1)    # (B, 5, 1000, 70) → (B, 64, 1000, 70)
        self.pool1 = nn.MaxPool2d(2, 2)        # → (B, 64, 500, 35)

        self.enc2 = ResConvBlock(c1, c2)       # → (B, 128, 500, 35)
        self.pool2 = nn.MaxPool2d(2, 2)        # → (B, 128, 250, 17)

        self.enc3 = ResConvBlock(c2, c3)       # → (B, 256, 250, 17)
        self.pool3 = nn.MaxPool2d(2, 2)        # → (B, 256, 125, 8)

        # ── BRIDGE: CNN → Transformer ──
        # Project to transformer dimension and compress spatial
        self.pre_bottleneck = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.GELU(),
        )
        # (B, 384, 125, 8)

        # ── SWIN TRANSFORMER BOTTLENECK ──
        # Global reasoning over the most compressed feature map
        self.bottleneck = SwinBottleneck(
            dim=c4,
            depth=bottleneck_depth,
            num_heads=bottleneck_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
        )

        # ── BRIDGE: Transformer → CNN ──
        self.post_bottleneck = nn.Sequential(
            nn.Conv2d(c4, c4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.GELU(),
        )

        # ── CNN DECODER ──
        # Each stage: Upsample + Concat skip + ResConvBlock (sharp reconstruction)
        self.up1 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2, output_padding=(0, 1))
        self.dec1 = ResConvBlock(c3 * 2, c3)   # 256 (up) + 256 (skip) → 256

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2, output_padding=(0, 1))
        self.dec2 = ResConvBlock(c2 * 2, c2)   # 128 (up) + 128 (skip) → 128

        self.up3 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec3 = ResConvBlock(c1 * 2, c1)   # 64 (up) + 64 (skip) → 64

        # ── OUTPUT HEAD ──
        self.output_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, out_ch, kernel_size=1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass with shape annotations at each stage.
        
        Args:
            x: (B, 5, 1000, 70)
        Returns:
            out: (B, 70, 70)
        """
        # ── CNN Encoder (local feature extraction) ──
        e1 = self.enc1(x)           # (B, 64, 1000, 70)
        p1 = self.pool1(e1)         # (B, 64, 500, 35)

        e2 = self.enc2(p1)          # (B, 128, 500, 35)
        p2 = self.pool2(e2)         # (B, 128, 250, 17)

        e3 = self.enc3(p2)          # (B, 256, 250, 17)
        p3 = self.pool3(e3)         # (B, 256, 125, 8)

        # ── Bridge → Transformer ──
        b = self.pre_bottleneck(p3)  # (B, 384, 125, 8)

        # ── Swin Bottleneck (global reasoning) ──
        b = self.bottleneck(b)       # (B, 384, 125, 8) — now globally aware

        # ── Bridge → CNN ──
        b = self.post_bottleneck(b)  # (B, 384, 125, 8)

        # ── CNN Decoder (sharp reconstruction) ──
        d1 = self.up1(b)                                # (B, 256, 250, 17)
        d1 = self._match_and_concat(d1, e3)             # handle size mismatch
        d1 = self.dec1(d1)                               # (B, 256, 250, 17)

        d2 = self.up2(d1)                                # (B, 128, 500, 35)
        d2 = self._match_and_concat(d2, e2)
        d2 = self.dec2(d2)                               # (B, 128, 500, 35)

        d3 = self.up3(d2)                                # (B, 64, 1000, 70)
        d3 = self._match_and_concat(d3, e1)
        d3 = self.dec3(d3)                               # (B, 64, 1000, 70)

        # ── Output ──
        out = self.output_head(d3)                       # (B, 1, 1000, 70)
        out = F.interpolate(out, size=(70, 70), mode="bilinear", align_corners=False)

        return out.squeeze(1)                            # (B, 70, 70)

    @staticmethod
    def _match_and_concat(upsampled, skip):
        """Crop/pad upsampled features to match skip connection spatial dims."""
        diff_h = skip.shape[2] - upsampled.shape[2]
        diff_w = skip.shape[3] - upsampled.shape[3]

        if diff_h != 0 or diff_w != 0:
            upsampled = F.pad(upsampled, [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2,
            ])

        return torch.cat([skip, upsampled], dim=1)


# ============================================================
# BACKWARD COMPATIBLE ALIAS
# ============================================================

def UNet(**kwargs):
    """Drop-in replacement — training script uses `from model import UNet`."""
    return HybridSwinUNet(**kwargs)


# ============================================================
# LOSS FUNCTIONS 
# ============================================================

def gradient_loss(pred, target):
    """Gradient difference loss — penalizes structural/edge mismatches."""
    dx_pred = pred[:, :, 1:] - pred[:, :, :-1]
    dx_true = target[:, :, 1:] - target[:, :, :-1]
    dy_pred = pred[:, 1:, :] - pred[:, :-1, :]
    dy_true = target[:, 1:, :] - target[:, :-1, :]
    return (dx_pred - dx_true).abs().mean() + (dy_pred - dy_true).abs().mean()


def ssim_loss(pred, target):
    """Structural Similarity (SSIM) loss."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = F.avg_pool2d(pred.unsqueeze(1), 3, 1, 1)
    mu_y = F.avg_pool2d(target.unsqueeze(1), 3, 1, 1)
    sigma_x = F.avg_pool2d(pred.unsqueeze(1) ** 2, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(target.unsqueeze(1) ** 2, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d((pred * target).unsqueeze(1), 3, 1, 1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return 1 - ssim.mean()


def tv_loss(pred):
    """Total Variation loss — light smoothness regularizer."""
    dx = pred[:, :, 1:] - pred[:, :, :-1]
    dy = pred[:, 1:, :] - pred[:, :-1, :]
    return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))


def physics_loss(pred_velocity, input_seismic):
    """Physics-informed loss using Deepwave forward modeling.

    Simulates wave propagation through the predicted velocity model
    and compares the synthetic receiver data to the actual input seismic.
    This forces the model to produce velocity maps that are physically
    consistent with the observed wavefield — shattering the 1D "safe zone."

    Args:
        pred_velocity: (B, 70, 70) — normalized velocity from model output
        input_seismic: (B, 5, 1000, 70) — seismic input (z-score normalized)

    Returns:
        Scalar physics consistency loss
    """
    import deepwave
    import config
    import random

    device = pred_velocity.device
    B = pred_velocity.shape[0]

    # De-normalize velocity to m/s and clamp to physical range
    v_ms = (pred_velocity * config.Y_MAX).clamp(min=1000.0, max=6000.0)

    # Ricker wavelet (same as OpenFWI generation)
    wavelet = deepwave.wavelets.ricker(
        config.FREQ, config.NT, config.DT, 1.0 / config.FREQ
    ).to(device)
    src_amp = wavelet.reshape(1, 1, -1)

    # Receiver locations: all 70 surface positions
    rec_loc = torch.zeros(1, config.N_RECEIVERS, 2, dtype=torch.long, device=device)
    rec_loc[0, :, 0] = config.RECEIVER_DEPTH
    rec_loc[0, :, 1] = torch.arange(config.N_RECEIVERS, device=device)

    total_loss = 0.0

    # Pick one random source index for this batch
    src_idx = random.randint(0, config.N_SOURCES - 1)
    src_x = config.SOURCE_POSITIONS[src_idx]

    src_loc = torch.zeros(1, 1, 2, dtype=torch.long, device=device)
    src_loc[0, 0, 0] = config.SOURCE_DEPTH
    src_loc[0, 0, 1] = src_x

    for b in range(B):
        v_single = v_ms[b]  # (70, 70)

        # Forward modeling through predicted velocity
        out = deepwave.scalar(
            v_single, grid_spacing=config.DX, dt=config.DT,
            source_amplitudes=src_amp,
            source_locations=src_loc,
            receiver_locations=rec_loc,
            pml_width=config.PML_WIDTH,
            pml_freq=config.FREQ,
            accuracy=4,
        )

        # Receiver data from Deepwave: last element of output tuple
        synthetic = out[-1].squeeze(0)  # (N_RECEIVERS, NT) or (NT, N_RECEIVERS)

        # Match shape to observed: (1000, 70) = (time, receivers)
        if synthetic.shape[0] == config.N_RECEIVERS:
            synthetic = synthetic.T  # (70, 1000) → (1000, 70)

        # Observed seismic for this source channel
        observed = input_seismic[b, src_idx]  # (1000, 70)

        # Normalize both to [-1, 1] (amplitude-invariant comparison)
        syn_norm = synthetic / (synthetic.abs().max() + 1e-8)
        obs_norm = observed / (observed.abs().max() + 1e-8)

        total_loss += F.mse_loss(syn_norm, obs_norm)

    return total_loss / B


# ============================================================
# VALIDATION
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    model = HybridSwinUNet(
        in_ch=5,
        base_ch=64,
        bottleneck_depth=6,
        bottleneck_heads=8,
        window_size=(5, 4),
        drop_path_rate=0.1,
    ).to(device)

    # ── Parameter count ──
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count by component
    enc_params = sum(
        p.numel() for name, p in model.named_parameters()
        if name.startswith("enc") or name.startswith("pool")
    )
    bot_params = sum(
        p.numel() for name, p in model.named_parameters()
        if "bottleneck" in name
    )
    dec_params = sum(
        p.numel() for name, p in model.named_parameters()
        if name.startswith("dec") or name.startswith("up") or name.startswith("output")
    )

    print(f"╔══════════════════════════════════════════╗")
    print(f"║   Hybrid CNN-Swin UNet V3.5              ║")
    print(f"╠══════════════════════════════════════════╣")
    print(f"║  Total params:     {total_params:>12,}       ║")
    print(f"║  Trainable params: {trainable_params:>12,}       ║")
    print(f"║                                          ║")
    print(f"║  CNN Encoder:      {enc_params:>12,}       ║")
    print(f"║  Swin Bottleneck:  {bot_params:>12,}       ║")
    print(f"║  CNN Decoder:      {dec_params:>12,}       ║")
    print(f"╚══════════════════════════════════════════╝")

    # ── DropPath info ──
    dp_modules = [m for m in model.modules() if isinstance(m, DropPath)]
    if dp_modules:
        rates = [m.drop_prob for m in dp_modules]
        print(f"\nDropPath: {len(dp_modules)} modules, "
              f"range [{min(rates):.3f} → {max(rates):.3f}]")

    # ── Forward pass test ──
    x = torch.randn(2, 5, 1000, 70).to(device)
    print(f"\nInput:  {x.shape}")

    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Output: {out.shape}")
    assert out.shape == (2, 70, 70), f"Expected (2, 70, 70), got {out.shape}"

    # ── Loss test ──
    target = torch.randn(2, 70, 70).to(device)
    losses = {
        "L1": F.l1_loss(out, target).item(),
        "MSE": F.mse_loss(out, target).item(),
        "Gradient": gradient_loss(out, target).item(),
        "SSIM": ssim_loss(out, target).item(),
        "TV": tv_loss(out).item(),
    }
    print(f"\nLosses:")
    for name, val in losses.items():
        print(f"  {name:>10}: {val:.4f}")

    # ── Memory estimate ──
    if device == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"\nGPU memory (inference, B=2): {mem_mb:.0f} MB")

    print(f"\n✅ Hybrid CNN-Swin UNet V3.5 validated!")