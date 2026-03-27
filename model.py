import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


# ============================================================
# DROP PATH (STOCHASTIC DEPTH)
# ============================================================

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample.
    
    During training, randomly drops entire residual branches with probability
    `drop_prob`. This acts as a regularizer — deeper layers get dropped more
    often, preventing overfitting and improving generalization.
    
    Args:
        x: input tensor of any shape
        drop_prob: probability of dropping the path (0.0 = no drop)
        training: whether model is in training mode
    Returns:
        scaled tensor (or zeros for dropped paths)
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    # Create random tensor with shape (batch_size, 1, 1, ...) for broadcasting
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor + keep_prob)
    # Scale surviving paths to maintain expected values
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample — as a module.
    
    Wraps the functional `drop_path` for use in nn.Sequential or as a layer.
    Automatically disabled during eval mode.
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


# ============================================================
# CORE SWIN TRANSFORMER COMPONENTS
# ============================================================

class MLP(nn.Module):
    """Feed-forward network used in each Swin block."""
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
    """Window-based Multi-Head Self-Attention (W-MSA / SW-MSA).
    
    Computes attention within local windows of size (window_h × window_w).
    Supports relative position bias for spatial awareness.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (wh, ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index for each token pair in the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, wh, ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, wh*ww)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # (N, N)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows * B, window_size * window_size, C)
            mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B_, num_heads, N, head_dim)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # (B_, num_heads, N, N)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """Partition feature map into non-overlapping windows.
    
    Args:
        x: (B, H, W, C)
        window_size: (wh, ww)
    Returns:
        windows: (num_windows * B, wh, ww, C)
    """
    B, H, W, C = x.shape
    wh, ww = window_size
    x = x.view(B, H // wh, wh, W // ww, ww, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, wh, ww, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition back to feature map.
    
    Args:
        windows: (num_windows * B, wh, ww, C)
        window_size: (wh, ww)
        H, W: original feature map dimensions
    Returns:
        x: (B, H, W, C)
    """
    wh, ww = window_size
    B = int(windows.shape[0] / (H * W / wh / ww))
    x = windows.view(B, H // wh, W // ww, wh, ww, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Single Swin Transformer block with optional shifted windows and DropPath.
    
    Even-indexed blocks use regular windows (W-MSA).
    Odd-indexed blocks use shifted windows (SW-MSA) for cross-window connections.
    
    DropPath is applied to BOTH residual branches (attention and MLP) to
    stochastically skip entire blocks during training.
    """
    def __init__(self, dim, num_heads, window_size=(7, 7), shift_size=(0, 0),
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path_rate=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop=drop)

        # DropPath on both residual connections
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial dimensions
        Returns:
            x: (B, H*W, C)
        """
        B, L, C = x.shape
        assert L == H * W, f"Input length {L} doesn't match H*W = {H}*{W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad feature map to be divisible by window size
        wh, ww = self.window_size
        pad_b = (wh - H % wh) % wh
        pad_r = (ww - W % ww) % ww
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = x.shape[1], x.shape[2]

        # Shifted window masking
        shift_h, shift_w = self.shift_size
        if shift_h > 0 or shift_w > 0:
            shifted_x = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(1, 2))
            attn_mask = self._compute_mask(Hp, Wp, x.device)
        else:
            shifted_x = x
            attn_mask = None

        # Partition into windows
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, wh, ww, C)
        x_windows = x_windows.view(-1, wh * ww, C)  # (nW*B, wh*ww, C)

        # Window attention
        attn_windows = self.attn(x_windows, mask=attn_mask)  # (nW*B, wh*ww, C)

        # Merge windows back
        attn_windows = attn_windows.view(-1, wh, ww, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse shift
        if shift_h > 0 or shift_w > 0:
            x = torch.roll(shifted_x, shifts=(shift_h, shift_w), dims=(1, 2))
        else:
            x = shifted_x

        # Remove padding
        if pad_b > 0 or pad_r > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # Residual + DropPath on attention branch
        x = shortcut + self.drop_path(x)

        # Residual + DropPath on MLP branch
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def _compute_mask(self, Hp, Wp, device):
        """Compute attention mask for shifted window self-attention."""
        wh, ww = self.window_size
        shift_h, shift_w = self.shift_size

        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (
            slice(0, -wh),
            slice(-wh, -shift_h),
            slice(-shift_h, None),
        )
        w_slices = (
            slice(0, -ww),
            slice(-ww, -shift_w),
            slice(-shift_w, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # (nW, wh, ww, 1)
        mask_windows = mask_windows.view(-1, wh * ww)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


class SwinTransformerStage(nn.Module):
    """A stage = multiple Swin Transformer blocks (alternating W-MSA and SW-MSA).
    
    DropPath rates increase linearly across blocks within the stage
    (deeper blocks are dropped more often — stochastic depth schedule).
    """
    def __init__(self, dim, depth, num_heads, window_size=(7, 7),
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path_rates=None):
        super().__init__()

        # Default: no drop path if rates not provided
        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth

        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = (0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2)
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim, num_heads=num_heads, window_size=window_size,
                    shift_size=shift, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                    drop_path_rate=drop_path_rates[i],
                )
            )

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        return x


# ============================================================
# PATCH OPERATIONS (Embedding, Merging, Expanding)
# ============================================================

class PatchEmbed(nn.Module):
    """Convert image to patch embeddings using convolution.
    
    Splits the input into non-overlapping patches and projects to embedding dim.
    """
    def __init__(self, in_ch=5, embed_dim=96, patch_size=(4, 4)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, H'*W', embed_dim), H', W'
        """
        x = self.proj(x)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', C)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    """Downsample by 2× — merges 2×2 patches, doubling channel dimension.
    
    Used between encoder stages (like MaxPool in CNN UNet).
    """
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C)
        Returns:
            x: (B, H/2 * W/2, 2*C), H//2, W//2
        """
        B, L, C = x.shape

        # Pad if H or W is odd
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = x.view(B, H, W, C)
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H, W = x.shape[1], x.shape[2]
            x = x.view(B, H * W, C)

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # top-left
        x1 = x[:, 1::2, 0::2, :]  # bottom-left
        x2 = x[:, 0::2, 1::2, :]  # top-right
        x3 = x[:, 1::2, 1::2, :]  # bottom-right
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)

        H, W = H // 2, W // 2
        x = x.view(B, H * W, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H, W


class PatchExpanding(nn.Module):
    """Upsample by 2× — expands patches, halving channel dimension.
    
    Used between decoder stages (like ConvTranspose in CNN UNet).
    """
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, dim * 2, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C)
        Returns:
            x: (B, 2H*2W, C/2), 2*H, 2*W
        """
        B, L, C = x.shape
        x = self.expand(x)  # (B, H*W, 2C)
        x = x.view(B, H, W, 2 * C)
        x = rearrange(x, "b h w (p1 p2 c) -> b (h p1) (w p2) c", p1=2, p2=2)
        H, W = H * 2, W * 2
        x = x.view(B, H * W, C // 2)
        x = self.norm(x)
        return x, H, W


class FinalPatchExpand(nn.Module):
    """Final 4× upsample to recover original patch resolution."""
    def __init__(self, dim, patch_size=(4, 4)):
        super().__init__()
        scale = patch_size[0] * patch_size[1]
        self.expand = nn.Linear(dim, dim * scale, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.patch_size = patch_size

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = self.expand(x)  # (B, H*W, C * p1 * p2)
        p1, p2 = self.patch_size
        x = x.view(B, H, W, p1, p2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * p1, W * p2, C)
        x = self.norm(x)
        return x  # (B, H*p1, W*p2, C)


# ============================================================
# SWIN-UNET MODEL
# ============================================================

class SwinUNet(nn.Module):
    """Swin-UNet: Pure Vision Transformer UNet with Stochastic Depth.
    
    Architecture:
        Input (B, 5, 1000, 70)
          → Patch Embed (4×2 patches)
          → Encoder: 3 stages with Patch Merging between them
          → Bottleneck: Swin Transformer stage
          → Decoder: 3 stages with Patch Expanding + skip connections
          → Final upsample + projection
          → Output (B, 70, 70)
    
    DropPath schedule: linearly increases from 0 to `drop_path_rate` across
    ALL blocks in the network (encoder + bottleneck + decoder). Deeper blocks
    have higher drop probability → acts as implicit ensemble regularizer.
    """
    def __init__(
        self,
        in_ch=5,
        out_ch=1,
        embed_dim=64,
        depths=(2, 2, 2, 2),         # blocks per stage: [enc1, enc2, enc3, bottleneck]
        num_heads=(2, 4, 8, 16),     # heads per stage
        window_size=(5, 5),           # window size for attention
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,           # stochastic depth rate
        patch_size=(4, 2),            # (4, 2) to handle 70-wide input cleanly
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depths = depths

        # ── Compute linearly increasing DropPath rates ──
        # Total blocks across ALL stages (encoder + bottleneck + decoder)
        total_encoder_blocks = sum(depths)
        total_decoder_blocks = sum(depths[:-1])  # decoder mirrors encoder (no bottleneck)
        total_blocks = total_encoder_blocks + total_decoder_blocks

        # Linear schedule: 0 → drop_path_rate across all blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Split rates for each stage
        # Encoder stages
        enc_block_idx = 0
        enc_dpr = []
        for d in depths:
            enc_dpr.append(dpr[enc_block_idx:enc_block_idx + d])
            enc_block_idx += d

        # Decoder stages (continue from where encoder left off)
        dec_block_idx = enc_block_idx
        dec_dpr = []
        for d in depths[:-1]:  # no bottleneck in decoder
            dec_dpr.append(dpr[dec_block_idx:dec_block_idx + d])
            dec_block_idx += d

        # ── Patch Embedding ──
        # (B, 5, 1000, 70) → (B, 250*35, 96) with H'=250, W'=35
        self.patch_embed = PatchEmbed(in_ch, embed_dim, patch_size)

        # ── ENCODER ──
        # Stage 1: dim=96, (250, 35)
        self.encoder1 = SwinTransformerStage(
            dim=embed_dim, depth=depths[0], num_heads=num_heads[0],
            window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path_rates=enc_dpr[0],
        )
        self.downsample1 = PatchMerging(embed_dim)

        # Stage 2: dim=192, (125, 18)
        self.encoder2 = SwinTransformerStage(
            dim=embed_dim * 2, depth=depths[1], num_heads=num_heads[1],
            window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path_rates=enc_dpr[1],
        )
        self.downsample2 = PatchMerging(embed_dim * 2)

        # Stage 3: dim=384, (63, 9)
        self.encoder3 = SwinTransformerStage(
            dim=embed_dim * 4, depth=depths[2], num_heads=num_heads[2],
            window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path_rates=enc_dpr[2],
        )
        self.downsample3 = PatchMerging(embed_dim * 4)

        # ── BOTTLENECK ──
        # dim=768, (32, 5)
        self.bottleneck = SwinTransformerStage(
            dim=embed_dim * 8, depth=depths[3], num_heads=num_heads[3],
            window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path_rates=enc_dpr[3],
        )

        # ── DECODER ──
        # Up 1: dim 768 → 384
        self.upsample1 = PatchExpanding(embed_dim * 8)
        self.skip_reduce1 = nn.Linear(embed_dim * 8, embed_dim * 4)
        self.decoder1 = SwinTransformerStage(
            dim=embed_dim * 4, depth=depths[2], num_heads=num_heads[2],
            window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path_rates=dec_dpr[2],
        )

        # Up 2: dim 384 → 192
        self.upsample2 = PatchExpanding(embed_dim * 4)
        self.skip_reduce2 = nn.Linear(embed_dim * 4, embed_dim * 2)
        self.decoder2 = SwinTransformerStage(
            dim=embed_dim * 2, depth=depths[1], num_heads=num_heads[1],
            window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path_rates=dec_dpr[1],
        )

        # Up 3: dim 192 → 96
        self.upsample3 = PatchExpanding(embed_dim * 2)
        self.skip_reduce3 = nn.Linear(embed_dim * 2, embed_dim)
        self.decoder3 = SwinTransformerStage(
            dim=embed_dim, depth=depths[0], num_heads=num_heads[0],
            window_size=window_size, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
            drop_path_rates=dec_dpr[0],
        )

        # ── Final Upsample + Output Head ──
        self.final_expand = FinalPatchExpand(embed_dim, patch_size)
        self.output_proj = nn.Conv2d(embed_dim, out_ch, kernel_size=1, bias=False)

        self.apply(self._init_weights)

        # Print drop path schedule
        self._print_drop_path_schedule(dpr, depths)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _print_drop_path_schedule(self, dpr, depths):
        """Print the stochastic depth schedule for debugging."""
        print("\n╔══════════════════════════════════════════════╗")
        print("║        DropPath (Stochastic Depth) Schedule  ║")
        print("╠══════════════════════════════════════════════╣")

        idx = 0
        stage_names = []
        for i, d in enumerate(depths[:-1]):
            stage_names.append(f"Encoder {i+1}")
        stage_names.append("Bottleneck")
        for i, d in enumerate(depths[:-1]):
            stage_names.append(f"Decoder {len(depths)-1-i}")

        all_depths = list(depths) + list(depths[:-1])

        for name, d in zip(stage_names, all_depths):
            rates = dpr[idx:idx + d]
            rates_str = ", ".join([f"{r:.3f}" for r in rates])
            print(f"║  {name:>12}: [{rates_str}]")
            idx += d

        print(f"║  {'Total blocks':>12}: {len(dpr)}")
        print(f"║  {'Max rate':>12}: {max(dpr):.3f}")
        print("╚══════════════════════════════════════════════╝\n")

    def _match_and_crop(self, upsampled, skip, target_H, target_W):
        """Crop or pad upsampled features to match skip connection dimensions.
        
        Because PatchMerging pads odd dimensions, the decoder must handle
        spatial mismatches between upsampled and skip features.
        """
        B = upsampled.shape[0]
        C_up = upsampled.shape[-1]
        C_skip = skip.shape[-1]

        up_H, up_W = target_H, target_W

        up_spatial = upsampled[:, :up_H * up_W, :].view(B, up_H, up_W, C_up)
        skip_spatial = skip[:, :target_H * target_W, :].view(B, target_H, target_W, C_skip)

        min_H = min(up_spatial.shape[1], skip_spatial.shape[1])
        min_W = min(up_spatial.shape[2], skip_spatial.shape[2])

        up_cropped = up_spatial[:, :min_H, :min_W, :]
        skip_cropped = skip_spatial[:, :min_H, :min_W, :]

        merged = torch.cat([up_cropped, skip_cropped], dim=-1)
        merged = merged.view(B, min_H * min_W, C_up + C_skip)

        return merged, min_H, min_W

    def forward(self, x):
        """
        Args:
            x: (B, 5, 1000, 70) — 5-channel seismic input
        Returns:
            out: (B, 70, 70) — velocity model prediction
        """
        B = x.shape[0]

        # ── Patch Embedding ──
        x, H, W = self.patch_embed(x)

        # ── Encoder ──
        enc1 = self.encoder1(x, H, W)
        enc1_H, enc1_W = H, W
        x, H, W = self.downsample1(enc1, H, W)

        enc2 = self.encoder2(x, H, W)
        enc2_H, enc2_W = H, W
        x, H, W = self.downsample2(enc2, H, W)

        enc3 = self.encoder3(x, H, W)
        enc3_H, enc3_W = H, W
        x, H, W = self.downsample3(enc3, H, W)

        # ── Bottleneck ──
        x = self.bottleneck(x, H, W)

        # ── Decoder ──
        x, H, W = self.upsample1(x, H, W)
        x, H, W = self._match_and_crop(x, enc3, min(H, enc3_H), min(W, enc3_W))
        x = self.skip_reduce1(x)
        x = self.decoder1(x, H, W)

        x, H, W = self.upsample2(x, H, W)
        x, H, W = self._match_and_crop(x, enc2, min(H, enc2_H), min(W, enc2_W))
        x = self.skip_reduce2(x)
        x = self.decoder2(x, H, W)

        x, H, W = self.upsample3(x, H, W)
        x, H, W = self._match_and_crop(x, enc1, min(H, enc1_H), min(W, enc1_W))
        x = self.skip_reduce3(x)
        x = self.decoder3(x, H, W)

        # ── Final Expand + Output ──
        x = self.final_expand(x, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.output_proj(x)
        x = F.interpolate(x, size=(70, 70), mode="bilinear", align_corners=False)

        return x.squeeze(1)


# ============================================================
# BACKWARD COMPATIBLE ALIAS
# ============================================================

def UNet(**kwargs):
    """Drop-in replacement: returns SwinUNet instead of CNN UNet.
    
    Training script can keep using `from model import UNet` unchanged.
    """
    return SwinUNet(**kwargs)


# ============================================================
# LOSS FUNCTIONS (unchanged)
# ============================================================

def gradient_loss(pred, target):
    """Gradient difference loss — penalizes structural/edge mismatches."""
    dx_pred = pred[:, :, 1:] - pred[:, :, :-1]
    dx_true = target[:, :, 1:] - target[:, :, :-1]
    dy_pred = pred[:, 1:, :] - pred[:, :-1, :]
    dy_true = target[:, 1:, :] - target[:, :-1, :]
    return (dx_pred - dx_true).abs().mean() + (dy_pred - dy_true).abs().mean()


def ssim_loss(pred, target):
    """Structural Similarity (SSIM) loss — measures perceptual quality."""
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
    """Total Variation loss — encourages smooth regions while preserving edges."""
    dx = pred[:, :, 1:] - pred[:, :, :-1]
    dy = pred[:, 1:, :] - pred[:, :-1, :]
    return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))


# ============================================================
# QUICK VALIDATION
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    model = SwinUNet(
        in_ch=5,
        embed_dim=64,
        depths=(2, 2, 2, 2),
        num_heads=(2, 4, 8, 16),
        window_size=(5, 5),
        patch_size=(4, 2),
        drop_path_rate=0.2,       # ← stochastic depth
        drop_rate=0.0,
        attn_drop_rate=0.0,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Count DropPath modules
    drop_path_modules = [m for m in model.modules() if isinstance(m, DropPath)]
    print(f"DropPath modules: {len(drop_path_modules)}")
    if drop_path_modules:
        rates = [m.drop_prob for m in drop_path_modules]
        print(f"  Min rate: {min(rates):.4f}")
        print(f"  Max rate: {max(rates):.4f}")
        print(f"  Mean rate: {sum(rates)/len(rates):.4f}")

    # Test forward pass — training mode (DropPath active)
    model.train()
    x = torch.randn(2, 5, 1000, 70).to(device)
    print(f"\nInput:  {x.shape}")

    with torch.no_grad():
        out_train = model(x)
    print(f"Output (train mode): {out_train.shape}")

    # Test forward pass — eval mode (DropPath disabled)
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    print(f"Output (eval mode):  {out_eval.shape}")

    # Verify DropPath makes a difference between train/eval
    diff = (out_train - out_eval).abs().mean().item()
    print(f"Train vs Eval diff:  {diff:.6f} (should be > 0)")

    assert out_train.shape == (2, 70, 70), f"Expected (2, 70, 70), got {out_train.shape}"
    assert out_eval.shape == (2, 70, 70), f"Expected (2, 70, 70), got {out_eval.shape}"
    print("\n✅ Forward pass successful!")

    # Test losses
    target = torch.randn(2, 70, 70).to(device)
    g_loss = gradient_loss(out_eval, target)
    s_loss = ssim_loss(out_eval, target)
    t_loss = tv_loss(out_eval)
    print(f"\nGradient loss: {g_loss.item():.4f}")
    print(f"SSIM loss:     {s_loss.item():.4f}")
    print(f"TV loss:       {t_loss.item():.4f}")
    print("\n✅ All losses computed successfully!")