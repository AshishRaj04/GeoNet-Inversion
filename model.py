import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # --- ENCODER ---
        # Input : (B, 5, 1000, 70)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        ) # Output: 32x1000x70

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 32x500x35

        self.enc2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        ) # Output: 64x500x35

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 64x250x17 (Note: 35/2 = 17.5 -> 17)

        # --- BOTTLENECK ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        ) # Output: 128x250x17


        # --- DECODER ---
        # Note: output_padding=(0, 1) fixes the dimension mismatch caused by pooling 35 into 17
        self.up1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, output_padding=(0, 1))
        # Output: 64x500x35

        # After concat with enc2, input channels = 64 (up1) + 64 (enc2) = 128
        self.dec1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        ) # Output: 64x500x35

        self.up2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        # Output: 32x1000x70

        # After concat with enc1, input channels = 32 (up2) + 32 (enc1) = 64
        self.dec2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        ) # Output: 32x1000x70

        # --- FINAL OUTPUT ---
        self.final = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        # Output: 1x1000x70

    def forward(self, x):
        # x : (B, 5, 1000, 70)

        # Encoder pass
        x1 = self.enc1(x)         # -> (B, 32, 1000, 70)
        x2 = self.pool1(x1)       # -> (B, 32, 500, 35)

        x3 = self.enc2(x2)        # -> (B, 64, 500, 35)
        x4 = self.pool2(x3)       # -> (B, 64, 250, 17)

        # Bottleneck
        x5 = self.bottleneck(x4)  # -> (B, 128, 250, 17)

        # Decoder pass 1
        x6 = self.up1(x5)         # -> (B, 64, 500, 35)
        # Skip connection 1 (concatenate along channel dimension)
        cat1 = torch.cat([x3, x6], dim=1) # -> (B, 128, 500, 35)
        x7 = self.dec1(cat1)      # -> (B, 64, 500, 35)

        # Decoder pass 2
        x8 = self.up2(x7)         # -> (B, 32, 1000, 70)
        # Skip connection 2
        cat2 = torch.cat([x1, x8], dim=1) # -> (B, 64, 1000, 70)
        x9 = self.dec2(cat2)      # -> (B, 32, 1000, 70)

        # Output projection
        out = self.final(x9)      # -> (B, 1, 1000, 70)

        # Spatial adjustment to match ground truth target grids
        out = nn.functional.interpolate(input=out, size=(70, 70), mode='bilinear', align_corners=False)
        # -> (B, 1, 70, 70)

        return out.squeeze(1)     # -> (B, 70, 70)

def gradient_loss(pred, target):
    dx_pred = pred[:, :, 1:] - pred[:, :, :-1]
    dx_true = target[:, :, 1:] - target[:, :, :-1]

    dy_pred = pred[:, 1:, :] - pred[:, :-1, :]
    dy_true = target[:, 1:, :] - target[:, :-1, :]

    return (dx_pred - dx_true).abs().mean() + (dy_pred - dy_true).abs().mean()

def ssim_loss(pred, target):
    C1 = 0.01**2
    C2 = 0.03**2

    mu_x = F.avg_pool2d(pred.unsqueeze(1), 3, 1, 1)
    mu_y = F.avg_pool2d(target.unsqueeze(1), 3, 1, 1)

    sigma_x = F.avg_pool2d(pred.unsqueeze(1)**2, 3, 1, 1) - mu_x**2
    sigma_y = F.avg_pool2d(target.unsqueeze(1)**2, 3, 1, 1) - mu_y**2
    sigma_xy = F.avg_pool2d((pred*target).unsqueeze(1), 3, 1, 1) - mu_x*mu_y

    ssim = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))

    return 1 - ssim.mean()