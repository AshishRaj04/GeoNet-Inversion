import modal
import os

app = modal.App("GeoNet-Evaluate")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0", "numpy", "matplotlib",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_python_source("config")
    .add_local_python_source("dataset")  
    .add_local_python_source("model")
    .add_local_python_source("utils")
)

vol_data = modal.Volume.from_name("FWI_Dataset", create_if_missing=False)
vol_result = modal.Volume.from_name("FWI_Result", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={"/dataset": vol_data, "/results_v5": vol_result},   
    secrets=[modal.Secret.from_name("FWI-Secret")],
    timeout=36000,
)
def evaluate_modal():
    import config
    import torch
    from torch.utils.data import DataLoader, Subset
    from model import UNet
    from dataset import MultiFileSeismicDataset   
    from utils import plot_prediction, calculate_metrics

    vol_data.reload()
    vol_result.reload()

    print(f"\nChecking data directory: {config.BASE_DATA_DIR}")
    if os.path.isdir(config.BASE_DATA_DIR):
        files = os.listdir(config.BASE_DATA_DIR)
        print(f"  Found {len(files)} files: {files[:5]}...")
    else:
        print(f"  ERROR: {config.BASE_DATA_DIR} does not exist!")
        print(f"  Contents of /data: {os.listdir('/data')}")
        raise FileNotFoundError(f"{config.BASE_DATA_DIR} not found")

    print(f"\nChecking model path: {config.MODEL_SAVE_PATH}")
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"  ✓ Model file found")
    else:
        print(f"  ✗ Model file NOT found")
        print(f"  Contents of {config.RESULTS_DIR}: {os.listdir(config.RESULTS_DIR) if os.path.isdir(config.RESULTS_DIR) else 'DIR NOT FOUND'}")
        raise FileNotFoundError(f"No trained model at {config.MODEL_SAVE_PATH}")

    device = config.DEVICE
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  GeoNet V5 — Modal Evaluation (Sample-Level Split)")
    print("=" * 60)

    print(f"\nLoading {len(config.ALL_FILE_PAIRS)} file pairs...")
    full_dataset = MultiFileSeismicDataset(config.ALL_FILE_PAIRS)
    total = len(full_dataset)

    generator = torch.Generator().manual_seed(42)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)

    indices = torch.randperm(total, generator=generator).tolist()
    test_idx = indices[train_size + val_size:]

    test_dataset = Subset(full_dataset, test_idx)
    loader = DataLoader(
        test_dataset, batch_size=1,
        num_workers=0, shuffle=False, pin_memory=True,  
    )

    print(f"  Total: {total} | Test samples: {len(test_dataset)}")
    uNet = UNet().to(device)
    uNet.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=True))
    print(f"  Loaded model from {config.MODEL_SAVE_PATH}")

    uNet.eval()
    print(f"  Parameters: {sum(p.numel() for p in uNet.parameters()):,}")

    all_metrics = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda"):
                pred = uNet(x)

            metrics = calculate_metrics(pred, y, config.Y_MAX)
            all_metrics.append(metrics)

            if i < 6:
                plot_prediction(
                    pred[0], y[0],
                    save_dir=config.RESULTS_DIR,
                    filename=f"pred_vs_GT_{i+1}.png",
                )

            if i < 10:
                print(
                    f"  Sample {i+1:>3} | MAE: {metrics['MAE']:>8.2f} | "
                    f"RMSE: {metrics['RMSE']:>8.2f} | SSIM: {metrics['SSIM']:.4f} | "
                    f"Edge: {metrics['Edge Error']:.4f}"
                )

    # Summary 
    avg_mae = sum(m["MAE"] for m in all_metrics) / len(all_metrics)
    avg_rmse = sum(m["RMSE"] for m in all_metrics) / len(all_metrics)
    avg_ssim = sum(m["SSIM"] for m in all_metrics) / len(all_metrics)
    avg_edge = sum(m["Edge Error"] for m in all_metrics) / len(all_metrics)

    print(f"\n{'=' * 50}")
    print(f"  Avg MAE:  {avg_mae:.2f} m/s")
    print(f"  Avg RMSE: {avg_rmse:.2f}")
    print(f"  Avg SSIM: {avg_ssim:.4f}")
    print(f"  Avg Edge: {avg_edge:.4f}")
    print(f"{'=' * 50}")

    # Save summary
    summary_path = os.path.join(config.RESULTS_DIR, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"GeoNet V5 Evaluation (Test Split — {len(test_dataset)} samples)\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Avg MAE:  {avg_mae:.2f} m/s\n")
        f.write(f"Avg RMSE: {avg_rmse:.2f}\n")
        f.write(f"Avg SSIM: {avg_ssim:.4f}\n")
        f.write(f"Avg Edge: {avg_edge:.4f}\n")
    print(f"  Summary saved to {summary_path}")
    vol_result.commit()

    return {
        "avg_mae": avg_mae,
        "avg_rmse": avg_rmse,
        "avg_ssim": avg_ssim,
        "avg_edge": avg_edge,
        "test_samples": len(test_dataset),
    }


@app.local_entrypoint()
def main():
    result = evaluate_modal.remote()
    print(f"\n🎯 Result: {result}")

    # Download evaluation results to local
    local_dir = os.path.join("results", "v5")
    os.makedirs(local_dir, exist_ok=True)

    print(f"\nDownloading evaluation results to {local_dir}...")
    for entry in vol_result.listdir("/"):
        remote_path = f"/{entry.path}"
        local_path = os.path.join(local_dir, os.path.basename(entry.path))
        try:
            with open(local_path, "wb") as f:
                for chunk in vol_result.read_file(remote_path):
                    f.write(chunk)
            print(f"  ↓ {entry.path}")
        except Exception as e:
            print(f"  ⚠ Skipping {entry.path}: {e}")

    print(f"✅ All evaluation files downloaded to {local_dir}")