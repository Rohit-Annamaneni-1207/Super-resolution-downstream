import math
import torch
import torch.nn as nn
# from torch.cuda import amp
from dataset import DIV2KPatchDataset
from tqdm import tqdm


torch.backends.cudnn.benchmark = True
    
class SRCNN(nn.Module):
    def __init__(self) -> None:
        super(SRCNN, self).__init__()
        # Feature extraction layer.
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True)
        )

        # Non-linear mapping layer.
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True)
        )

        # Rebuild the layer.
        self.reconstruction = nn.Conv2d(32, 1, (5, 5), (1, 1), (2, 2))

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)

        return out
    
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())))
                nn.init.zeros_(module.bias.data)

        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)
    
    
def validate_mse(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for lr_up, hr in loader:
            lr_up = lr_up.to(device)
            hr = hr.to(device)

            sr = model(lr_up)
            loss = criterion(sr, hr)
            total_loss += loss.item()
    return total_loss / len(loader)


def train_srcnn():

    scale = 3
    batch_size = 32

    # Your folders
    hr_dir = r"D:\DIP Project\\Train\\DIV2K_train_HR"
    lr_dir = r"D:\DIP Project\\Train\DIV2K_train_LR_unknown\\X3"

    # Dataset + DataLoader
    dataset = DIV2KPatchDataset(hr_dir, lr_dir, scale=scale)
    print(f"Number of samples in train dataset: {len(dataset)}")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    val_hr_dir = r"D:\DIP Project\\Val\\DIV2K_valid_HR"
    val_lr_dir = r"D:\DIP Project\\Val\\DIV2K_valid_LR_unknown\\X3"

    val_dataset = DIV2KPatchDataset(val_hr_dir, val_lr_dir, scale=scale, mode='val')
    print(f"Number of samples in val dataset: {len(val_dataset)}")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)


    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = SRCNN().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 50
    best_val_loss = float("inf")
    patience = 0
    max_patience = 10

    print("Starting training...")


    for epoch in range(0, epochs):
        model.train(True)
        epoch_loss = 0.0

        loop = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=True)

        for lr_up, hr in loop:
            lr_up = lr_up.to(device)
            hr = hr.to(device)

            optimizer.zero_grad()
            
            sr = model(lr_up)
            loss = criterion(sr, hr)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch} Completed | Avg Loss: {epoch_loss/len(loader):.6f}")

        val_loss = validate_mse(model, val_loader, criterion, device)
        print(f"Validation MSE after Epoch {epoch}: {val_loss:.6f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), f"SRCNN_best_MSE_x{scale}_patched_opt.pth")
            print(f"New best model saved! MSE = {best_val_loss:.6f}")
        else:
            patience += 1

        # if patience >= max_patience:
        #     print(f"Early stopping: no improvement for {max_patience} epochs.")
        #     break

    return model

if __name__ == "__main__":
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    model = train_srcnn()

