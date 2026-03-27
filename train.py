import modal
import os

app = modal.App("GeoNet-Inversion")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "matplotlib", "einops")
    .add_local_python_source("config")
    .add_local_python_source("dataset")   
    .add_local_python_source("model")
    .add_local_python_source("utils")
    .add_local_python_source("train")
    .add_local_python_source("evaluate")
)

vol_data = modal.Volume.from_name("FWI_Dataset", create_if_missing=True)
vol_result = modal.Volume.from_name("FWI_Result", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/dataset": vol_data, "/results_v3": vol_result},   
    secrets=[modal.Secret.from_name("FWI-Secret")],
    timeout=36000,
)
def train():
    import config
    import torch
    from torch.utils.data import DataLoader, Subset
    from model import UNet, gradient_loss, ssim_loss, tv_loss
    from dataset import MultiFileSeismicDataset       
    from utils import plot_loss, plot_lrs

    # Reload volume to ensure latest data is visible 
    vol_data.reload()

    # Debug: verify files exist 
    print(f"\nChecking data directory: {config.BASE_DATA_DIR}")
    if os.path.isdir(config.BASE_DATA_DIR):
        files = os.listdir(config.BASE_DATA_DIR)
        print(f"  Found {len(files)}...")
    else:
        print(f"  ERROR: {config.BASE_DATA_DIR} does not exist!")
        print(f"  Contents of /dataset: {os.listdir('/dataset')}")
        raise FileNotFoundError(f"{config.BASE_DATA_DIR} not found")

    device = config.DEVICE
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # ── Sample-level split ──
    print("=" * 60)
    print("  GeoNet V3")
    print("=" * 60)

    print(f"\nLoading {len(config.ALL_FILE_PAIRS)} file pairs...")
    full_dataset = MultiFileSeismicDataset(config.ALL_FILE_PAIRS)
    total = len(full_dataset)
    print(f"Total samples: {total}")

    generator = torch.Generator().manual_seed(42)
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)

    indices = torch.randperm(total, generator=generator).tolist()
    train_dataset = Subset(full_dataset, indices[:train_size])
    val_dataset = Subset(full_dataset, indices[train_size:train_size + val_size])

    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {total - train_size - val_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        num_workers=0, shuffle=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        num_workers=0, shuffle=False, pin_memory=True,
    )

    # ── Model ──
    uNet = UNet().to(device)
    total_params = sum(p.numel() for p in uNet.parameters())
    print(f"\nModel: {total_params:,} params | Device: {device}")
    print(f"Batch: {config.BATCH_SIZE} | Epochs: {config.EPOCHS} | LR: {config.LEARNING_RATE}")

    optimizer = torch.optim.AdamW(uNet.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.WARMUP_EPOCHS,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS - config.WARMUP_EPOCHS, eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.WARMUP_EPOCHS],
    )

    l1 = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()

    # Mixed precision scaler
    scaler = torch.amp.GradScaler("cuda")

    train_losses = []
    val_losses = []
    lrs = []
    best_val_loss = float("inf")

    ckpt_path = os.path.join(config.RESULTS_DIR, "checkpoint_epoch_150.pth")
    start_epoch = 0
    
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        uNet.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] 
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint["val_losses"]
        lrs = checkpoint["lrs"]
        if val_losses:
            best_val_loss = min(val_losses)
        print(f"Resuming training from epoch {start_epoch}. Best Val Loss so far: {best_val_loss:.4f}")

    print(f"\n{'Epoch':>6} | {'Train Loss':>11} | {'Val Loss':>11} | {'Val SSIM':>9} | {'LR':>10}")
    print("-" * 65)

    # ── Training loop ──
    for epoch in range(start_epoch, config.EPOCHS):
        uNet.train()
        total_train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                y_pred = uNet(x_batch)
                loss = config.TOTAL_LOSS_SCALE * (
                    config.L1_WEIGHT * l1(y_pred, y_batch)
                    + config.MSE_WEIGHT * mse(y_pred, y_batch)
                    + config.GRADIENT_WEIGHT * gradient_loss(y_pred, y_batch)
                    + config.SSIM_WEIGHT * ssim_loss(y_pred, y_batch)
                    + config.TV_WEIGHT * tv_loss(y_pred)
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(uNet.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        mean_train_loss = total_train_loss / len(train_loader)
        train_losses.append(mean_train_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        scheduler.step()

        uNet.eval()
        total_val_loss = 0.0
        total_val_ssim = 0.0
        val_batches = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                with torch.amp.autocast("cuda"):
                    y_pred = uNet(x_batch)
                    vloss = config.TOTAL_LOSS_SCALE * (
                        config.L1_WEIGHT * l1(y_pred, y_batch)
                        + config.MSE_WEIGHT * mse(y_pred, y_batch)
                        + config.GRADIENT_WEIGHT * gradient_loss(y_pred, y_batch)
                        + config.SSIM_WEIGHT * ssim_loss(y_pred, y_batch)
                        + config.TV_WEIGHT * tv_loss(y_pred)
                    )

                total_val_loss += vloss.item()
                total_val_ssim += (1 - ssim_loss(y_pred, y_batch)).item()
                val_batches += 1

        mean_val_loss = total_val_loss / len(val_loader)
        mean_val_ssim = total_val_ssim / val_batches
        val_losses.append(mean_val_loss)      

        # Print at VAL_FREQ intervals
        if (epoch + 1) % config.VAL_FREQ == 0 or epoch == 0:
            print(
                f"{epoch+1:>6} | {mean_train_loss:>11.4f} | {mean_val_loss:>11.4f} "
                f"| {mean_val_ssim:>9.4f} | {current_lr:>10.6f}"
            )

        # Save best model
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(uNet.state_dict(), config.MODEL_SAVE_PATH)
            vol_result.commit()
            print(f"         ✓ Best model saved (val_loss: {best_val_loss:.4f})")

        # ── Checkpoint ──
        if (epoch + 1) % config.CHECKPOINT_FREQ == 0:
            ckpt_path = os.path.join(config.RESULTS_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": uNet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "lrs": lrs,
                },
                ckpt_path,
            )
            vol_result.commit()
            print(f"         ✓ Checkpoint saved (epoch {epoch+1})")

    # ── Post-training ──
    print("\n" + "=" * 60)
    print(f"  Training complete! Best val loss: {best_val_loss:.4f}")
    print("=" * 60)

    plot_loss(train_losses, val_losses, save_dir=config.RESULTS_DIR)
    plot_lrs(lrs, save_dir=config.RESULTS_DIR)

    torch.save(
        {
            "model_state_dict": uNet.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "lrs": lrs,
            "best_val_loss": best_val_loss,
        },
        os.path.join(config.RESULTS_DIR, "final_training_state.pth"),
    )
    vol_result.commit()

    return {"best_val_loss": best_val_loss, "final_train_loss": train_losses[-1]}


@app.local_entrypoint()
def main():
    result = train.remote()
    print(f"\n🎯 Result: {result}")

    # Download results from volume to local
    local_dir = os.path.join("results", "v3")
    os.makedirs(local_dir, exist_ok=True)

    print(f"\nDownloading results to {local_dir}...")
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

    print(f"✅ All files downloaded to {local_dir}")