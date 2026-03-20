import torch
import argparse
import os
from torch.utils.data import DataLoader
from model import UNet
from dataset import SeismicDataset
from utils import plot_prediction, calculate_metrics
import config

def evaluate(untrained=False):
    device = config.DEVICE
    
    dataset = SeismicDataset(config.SEISMIC_DATA_PATH, config.VELOCITY_DATA_PATH)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    y_max = dataset.y_max

    uNet = UNet().to(device)

    if not untrained:
        if os.path.exists(config.MODEL_SAVE_PATH):
            uNet.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
            print(f"Loaded trained model from {config.MODEL_SAVE_PATH}")
        else:
            print("No trained model found. Running with untrained weights.")
    else:
        print("Running with untrained model for sanity check.")

    uNet.eval()
    print(f"Number of parameters: {sum(param.numel() for param in uNet.parameters())}")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = uNet(x)

            metrics = calculate_metrics(pred, y, y_max)
            
            print(f"Metrics:")
            for name, val in metrics.items():
                print(f"  {name}: {val:.4f}")

            plot_prediction(pred[0], y[0])
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--untrained", action="store_true", help="Run sanity check with untrained model")
    args = parser.parse_args()
    
    evaluate(untrained=args.untrained)

# Rough targets:
# MAE   < 300–400 m/s   ✅
# RMSE  < 500–700       ✅
# SSIM  > 0.85          ✅
# Edge  → as low as possible