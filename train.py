import torch
from torch.utils.data import DataLoader
from model import UNet, gradient_loss, ssim_loss
from dataset import SeismicDataset
import config

def train():
    device = config.DEVICE
    
    train_dataset = SeismicDataset(config.SEISMIC_DATA_PATH, config.VELOCITY_DATA_PATH)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        num_workers=0, 
        shuffle=True
    )

    uNet = UNet().to(device)

    optimizer = torch.optim.AdamW(uNet.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10
    )

    l1 = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()

    losses = []

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
        scheduler.step(mean_loss)  

        print(f"Epoch : {epoch+1}/{config.EPOCHS}, Loss : {mean_loss:.4f}")

    torch.save(uNet.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()