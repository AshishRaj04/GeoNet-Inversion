import config
import torch
from torch.utils.data import DataLoader
from model import UNet, gradient_loss, ssim_loss
from dataset import SeismicDataset
from utils import plot_loss, plot_lrs
import config

def train():
    device = config.DEVICE
    
    train_dataset = SeismicDataset(config.SEISMIC_DATA_PATH, config.VELOCITY_DATA_PATH)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        num_workers=2, 
        shuffle=True
    )

    uNet = UNet().to(device)

    optimizer = torch.optim.AdamW(uNet.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=100)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS - 100,
        eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[100]
    )

    total_parameters = 0
    for param in uNet.parameters():
        total_parameters += param.numel()
    print(f"Total parameters: {total_parameters}") 

    l1 = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()

    losses = []
    lrs = []

    for epoch in range(config.EPOCHS):
        uNet.train()
        total_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            y_pred = uNet(x_batch)

            loss = config.TOTAL_LOSS_SCALE * (
                config.L1_WEIGHT * l1(y_pred, y_batch)
                + config.MSE_WEIGHT * mse(y_pred, y_batch)
                + config.GRADIENT_WEIGHT * gradient_loss(y_pred, y_batch)
                + config.SSIM_WEIGHT * ssim_loss(y_pred, y_batch)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        mean_loss = total_loss / len(train_loader)
        losses.append(mean_loss)
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

        print(f"Epoch : {epoch+1}/{config.EPOCHS}, Loss : {mean_loss:.4f}")

    torch.save(uNet.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")
    plot_loss(losses)
    plot_lrs(lrs)

if __name__ == "__main__":
    train()