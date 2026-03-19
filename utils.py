import torch
import matplotlib.pyplot as plt
import numpy as np
from model import ssim_loss

def plot_prediction(y_pred, y_true):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    if y_pred.ndim == 3:
        y_pred = y_pred.squeeze()
    if y_true.ndim == 3:
        y_true = y_true.squeeze()

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Prediction")
    plt.imshow(y_pred, cmap='jet')
    plt.colorbar(label='Velocity (Normalized)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.subplot(1, 2, 2)
    plt.title("Ground Truth")
    plt.imshow(y_true, cmap='jet')
    plt.colorbar(label='Velocity (Normalized)')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.tight_layout()
    plt.savefig("results/pred_vs_GT.png", format="png")
    plt.show()

def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title("Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/loss_vs_epochs.png", format="png")
    plt.show()

def mae_physical(pred, target, y_max):
    p = pred * y_max
    t = target * y_max
    return torch.mean(torch.abs(p - t))

def rmse(pred, target, y_max):
    p = pred * y_max
    t = target * y_max
    return torch.sqrt(torch.mean((p - t) ** 2))

def edge_error(pred, target):
    dx_p = pred[:, :, 1:] - pred[:, :, :-1]
    dx_t = target[:, :, 1:] - target[:, :, :-1]

    dy_p = pred[:, 1:, :] - pred[:, :-1, :]
    dy_t = target[:, 1:, :] - target[:, :-1, :]

    return torch.mean(torch.abs(dx_p - dx_t)) + torch.mean(torch.abs(dy_p - dy_t))

def calculate_metrics(pred, target, y_max):
    metrics = {
        "MAE": mae_physical(pred, target, y_max).item(),
        "RMSE": rmse(pred, target, y_max).item(),
        "SSIM": (1 - ssim_loss(pred, target)).item(),
        "Edge Error": edge_error(pred, target).item()
    }
    return metrics
