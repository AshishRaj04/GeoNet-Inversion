import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# BUILDING BLOCKS
# ============================================================

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → LeakyReLU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Residual block: two convolutions with skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class UpBlock(nn.Module):
    """Upsample → Conv → BN → LeakyReLU → ResBlock"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, 3, 1, 1)
        self.res = ResBlock(out_ch)

    def forward(self, x, target_size):
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = self.res(x)
        return x


# ============================================================
# V5: DEEP CNN ENCODER-DECODER (InversionNet-style)
# ============================================================

class UNet(nn.Module):
    """Deep CNN encoder-decoder for seismic-to-velocity inversion.

    Architecture philosophy:
    - Deep encoder with asymmetric strides to compress the time dimension
      (1000 → ~8) while preserving spatial info (70 → ~9)
    - Wide bottleneck: (512, 8, 9) = 36,864 values (20x bigger than V4)
    - Decoder upsamples back to (70, 70) velocity map
    - No transformers, no skip connections between encoder/decoder
      (seismic time-domain ≠ velocity depth-domain)
    - Residual blocks throughout for stable deep training

    Input:  (B, 5, 1000, 70) — 5 seismic shot gathers
    Output: (B, 70, 70)      — velocity map (normalized)
    """

    def __init__(self):
        super().__init__()

        # ─── Encoder ───
        # Phase 1: Compress time dimension (stride only in time axis)
        # (5, 1000, 70) → (64, 250, 70)
        self.enc1 = ConvBlock(5, 64, kernel_size=(7, 3), stride=(4, 1), padding=(3, 1))
        self.enc1_res = ResBlock(64)

        # (64, 250, 70) → (128, 63, 70)
        self.enc2 = ConvBlock(64, 128, kernel_size=(5, 3), stride=(4, 1), padding=(2, 1))
        self.enc2_res = ResBlock(128)

        # Phase 2: Symmetric compression (both dimensions)
        # (128, 63, 70) → (256, 32, 35)
        self.enc3 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc3_res = ResBlock(256)

        # (256, 32, 35) → (512, 16, 18)
        self.enc4 = ConvBlock(256, 512, kernel_size=3, stride=2, padding=1)
        self.enc4_res = ResBlock(512)

        # (512, 16, 18) → (512, 8, 9)
        self.enc5 = ConvBlock(512, 512, kernel_size=3, stride=2, padding=1)
        self.enc5_res = ResBlock(512)

        # ─── Bottleneck ───
        # (512, 8, 9) = 36,864 values — wide enough to preserve lateral structure
        self.bottleneck1 = ResBlock(512)
        self.bottleneck2 = ResBlock(512)

        # ─── Decoder ───
        # Upsample progressively to (70, 70)
        self.dec1 = UpBlock(512, 256)   # → (256, 16, 18)
        self.dec2 = UpBlock(256, 128)   # → (128, 35, 35)
        self.dec3 = UpBlock(128, 64)    # → (64, 70, 70)

        # Output head — refine and project to 1 channel
        self.refine = ResBlock(64)
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        # x: (B, 5, 1000, 70)

        # ─── Encode ───
        x = self.enc1_res(self.enc1(x))    # (B, 64, 250, 70)
        x = self.enc2_res(self.enc2(x))    # (B, 128, 63, 70)
        x = self.enc3_res(self.enc3(x))    # (B, 256, 32, 35)
        x = self.enc4_res(self.enc4(x))    # (B, 512, 16, 18)
        x = self.enc5_res(self.enc5(x))    # (B, 512, 8, 9)

        # ─── Bottleneck ───
        x = self.bottleneck1(x)            # (B, 512, 8, 9)
        x = self.bottleneck2(x)            # (B, 512, 8, 9)

        # ─── Decode ───
        x = self.dec1(x, (16, 18))         # (B, 256, 16, 18)
        x = self.dec2(x, (35, 35))         # (B, 128, 35, 35)
        x = self.dec3(x, (70, 70))         # (B, 64, 70, 70)

        # ─── Output ───
        x = self.refine(x)                 # (B, 64, 70, 70)
        x = self.head(x)                   # (B, 1, 70, 70)
        x = x.squeeze(1)                   # (B, 70, 70)
        return x


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class SSIM(nn.Module):
    """Structural Similarity Index for single-channel images."""
    def __init__(self, window_size=7, sigma=1.5, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.C1 = C1
        self.C2 = C2
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel = g.unsqueeze(0) * g.unsqueeze(1)
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel[None, None, :, :])

    def forward(self, x, y):
        x = x.float()
        y = y.float()
        x = x.unsqueeze(1) if x.ndim == 3 else x
        y = y.unsqueeze(1) if y.ndim == 3 else y
        k = self.kernel.to(x.device)
        pad = k.shape[-1] // 2
        mu_x = F.conv2d(x, k, padding=pad)
        mu_y = F.conv2d(y, k, padding=pad)
        s_xx = F.conv2d(x * x, k, padding=pad) - mu_x * mu_x
        s_yy = F.conv2d(y * y, k, padding=pad) - mu_y * mu_y
        s_xy = F.conv2d(x * y, k, padding=pad) - mu_x * mu_y
        num = (2 * mu_x * mu_y + self.C1) * (2 * s_xy + self.C2)
        den = (mu_x**2 + mu_y**2 + self.C1) * (s_xx + s_yy + self.C2)
        return (num / den).mean()


_ssim_module = None

def ssim_loss(pred, target):
    """1 - SSIM (differentiable loss)."""
    global _ssim_module
    if _ssim_module is None:
        _ssim_module = SSIM().to(pred.device)
    return 1 - _ssim_module(pred, target)


def gradient_loss(pred, target):
    """Sobel-like gradient matching loss."""
    pred_dx = pred[:, :, 1:] - pred[:, :, :-1]
    pred_dy = pred[:, 1:, :] - pred[:, :-1, :]
    tgt_dx = target[:, :, 1:] - target[:, :, :-1]
    tgt_dy = target[:, 1:, :] - target[:, :-1, :]
    return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)


def tv_loss(pred):
    """Total Variation loss — light smoothness regularizer."""
    dx = pred[:, :, 1:] - pred[:, :, :-1]
    dy = pred[:, 1:, :] - pred[:, :-1, :]
    return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))


# ============================================================
# VALIDATION
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    model = UNet().to(device)

    # ── Parameter count ──
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    enc_params = sum(
        p.numel() for name, p in model.named_parameters()
        if name.startswith("enc")
    )
    bot_params = sum(
        p.numel() for name, p in model.named_parameters()
        if "bottleneck" in name
    )
    dec_params = sum(
        p.numel() for name, p in model.named_parameters()
        if name.startswith("dec") or name.startswith("refine") or name.startswith("head")
    )

    print(f"╔══════════════════════════════════════════╗")
    print(f"║   Deep CNN Encoder-Decoder V5            ║")
    print(f"╠══════════════════════════════════════════╣")
    print(f"║  Total params:     {total_params:>12,}       ║")
    print(f"║  Trainable params: {trainable_params:>12,}       ║")
    print(f"║                                          ║")
    print(f"║  CNN Encoder:      {enc_params:>12,}       ║")
    print(f"║  Bottleneck:       {bot_params:>12,}       ║")
    print(f"║  CNN Decoder:      {dec_params:>12,}       ║")
    print(f"╚══════════════════════════════════════════╝")

    # ── Forward pass test ──
    seismic_path = "data/FlatFault_A_seis2_1_0.npy"
    seismic_np = np.load(seismic_path)
    x = torch.from_numpy(seismic_np[0:2]).float().to(device)
    print(f"\nInput:  {x.shape}")

    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Output: {out.shape}")
    assert out.shape == (2, 70, 70), f"Expected (2, 70, 70), got {out.shape}"

    # ── Loss test ──
    velocity_path = "data/FlatFault_A_vel2_1_0.npy"
    vel_np = np.load(velocity_path)
    target = torch.from_numpy(vel_np[0:2]).float().squeeze(1).to(device)
    target = target / target.max()
    out = out / out.max()
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

    print(f"\n✅ Deep CNN Encoder-Decoder V5 validated!")