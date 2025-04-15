import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import logging, time, tqdm, sys
import argparse

# https://interdigitalinc.github.io/CompressAI/_modules/compressai/losses/rate_distortion.html#RateDistortionLoss
from compressai.losses import RateDistortionLoss
from compressai.entropy_models import EntropyBottleneck

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s - %(asctime)s', datefmt='%H:%M:%S')

IMAGE_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.001

NUM_EPOCHS = 250
BATCH_SIZE = 128
SAVE_INTER = 50

HIDDEN_DIM = 256
LATENT_DIM = 1024

NUM_CHANNELS = 3
CONV_FILTERS_1 = 32 
CONV_FILTERS_2 = 64
CONV_FILTERS_3 = 128
CONV_FILTERS_4 = 256 

BOTTLENECK_FILTERS = 64

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # Dummy label (not used in VAE)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = nn.Sequential(
        #     # convolutional layers
        #     nn.Conv2d(NUM_CHANNELS, CONV_FILTERS_1, kernel_size=3, padding=1, stride=2), 
        #     nn.BatchNorm2d(CONV_FILTERS_1),
        #     nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(2, 2),
        #     nn.Conv2d(CONV_FILTERS_1, CONV_FILTERS_2, kernel_size=3, padding=1, stride=2), 
        #     nn.BatchNorm2d(CONV_FILTERS_2),
        #     nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(2, 2),
            
        #     nn.Conv2d(CONV_FILTERS_2, CONV_FILTERS_3, kernel_size=3, padding=1, stride=2), 
        #     nn.BatchNorm2d(CONV_FILTERS_3),
        #     nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(2, 2),
        #     nn.Conv2d(CONV_FILTERS_3, CONV_FILTERS_4, kernel_size=3, padding=1, stride=2),
        #     nn.BatchNorm2d(CONV_FILTERS_4),
        #     nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(2, 2)

        #     # bottleneck
        #     nn.Conv2d(CONV_FILTERS_4, BOTTLENECK_FILTERS, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(BOTTLENECK_FILTERS),
        #     nn.ReLU(inplace=True),
        # )
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2),  # (B, 64, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),  # (B, 128, H/4, W/4)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 5, stride=2, padding=2),  # (B, 256, H/8, W/8)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 1), 
        )     

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.decoder = nn.Sequential(
        #     # bottleneck
        #     nn.ConvTranspose2d(BOTTLENECK_FILTERS, CONV_FILTERS_4, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(CONV_FILTERS_4),
        #     nn.ReLU(inplace=True),

        #     nn.ConvTranspose2d(CONV_FILTERS_4, CONV_FILTERS_3, kernel_size=3, stride=2, padding=1, output_padding=1), 
        #     nn.BatchNorm2d(CONV_FILTERS_3),
        #     nn.ReLU(inplace=True),
            
        #     nn.ConvTranspose2d(CONV_FILTERS_3, CONV_FILTERS_2, kernel_size=3, stride=2, padding=1, output_padding=1), 
        #     nn.BatchNorm2d(CONV_FILTERS_2),
        #     nn.ReLU(inplace=True),
            
        #     nn.ConvTranspose2d(CONV_FILTERS_2, CONV_FILTERS_1, kernel_size=3, stride=2, padding=1, output_padding=1), 
        #     nn.BatchNorm2d(CONV_FILTERS_1),
        #     nn.ReLU(inplace=True),
            
        #     nn.ConvTranspose2d(CONV_FILTERS_1, NUM_CHANNELS, kernel_size=3, stride=2, padding=1, output_padding=1), 
        #     nn.Sigmoid()
        # )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 256, 4, stride=2, padding=1),  # (B, 256, H/4, W/4)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # (B, 128, H/2, W/2)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # (B, 64, H, W)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()  # output in [0, 1]
        )

    def forward(self, x):
        return self.decoder(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(64).to(DEVICE)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        # encoded = self.encoder(x)
        # # Apply entropy bottleneck (real entropy bottleneck operation)
        # encoded, likelihoods = self.entropy_bottleneck(encoded)
        # decoded = self.decoder(encoded)

        # residual = x - decoded
        # return decoded, residual

        encoded = self.encoder(x)
        # Apply entropy bottleneck (real entropy bottleneck operation)
        encoded, likelihoods = self.entropy_bottleneck(encoded)
        decoded = self.decoder(encoded)

        residual = x - decoded
        return {"x_hat": decoded, "likelihoods": {"y": likelihoods}}


def autoencoder_loss(x, recon_x, residual):
    recon_loss = F.mse_loss(recon_x, x)
    residual_loss = F.mse_loss(residual, torch.zeros_like(residual))
    return recon_loss + residual_loss

def measure_compression(model, data_loader):
    data_point = data_loader.dataset[0][0].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logging.debug(f"DATA SIZE: {data_point.shape}")
        encoded = model.encoder(data_point)
        logging.debug(f"ENCODED SIZE: {encoded.shape}")
        compression_ratio = data_point.numel() / encoded.numel()
        logging.debug(f"Facteur de compression: {compression_ratio:.2f}")

def train(train_loader, valid_loader, save_path, usesWandb=False):
    model = Autoencoder().to(DEVICE)
    # model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    # NUM_EPOCHS = 50
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = RateDistortionLoss(lmbda=0.01, metric="mse", return_type="loss")
    # 10 epochs sans amélioration on diminue le learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    if usesWandb:
        import wandb
        wandb.init(
            entity="zevictos-polytechnique-montreal",
            # Set the wandb project where this run will be logged.
            project="Projet final",
            # Track hyperparameters and run metadata.
            config={
                "learning_rate": 0.02,
                "architecture": "Autoencoder",
                "dataset": "dogs",
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "image_size": IMAGE_SIZE,
            },
        )
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        measure_compression(model, train_loader)
        with tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch", position=0, leave=False, ncols=100) as t_loader:
            for data, _ in t_loader:
                data = data.to(DEVICE)
                optimizer.zero_grad()
            
                # recon_batch, residual_batch = model(data)
                recon_batch = model(data)
                # loss = autoencoder_loss(data, recon_batch, residual_batch)
                loss = criterion(recon_batch, data)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                   # Log batch loss to wandb (optional, can log per epoch instead)
                if usesWandb:
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": loss.item(),
                        # "running_train_loss": train_loss / (batch_idx + 1),
                    })
        
                    # Log average epoch loss
                    avg_train_loss = train_loss / len(train_loader)
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                    })
        scheduler.step(train_loss)
        
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, _ in valid_loader:
                inputs = inputs.to(DEVICE)
                # decoded, residual = model(inputs)
                decoded = model(inputs)
                loss = criterion(decoded, inputs)
                # loss = autoencoder_loss(inputs, decoded, residual)
                val_loss += loss.item()

        val_loss /= len(valid_loader)
        train_loss /= len(train_loader)
        scheduler.step(val_loss)

        logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")
        
        if (epoch + 1) % 50 == 0:
            show_image(model, valid_loader)
        if (epoch + 1) % SAVE_INTER == 0:
            torch.save(model.state_dict(), os.path.join(save_path, os.path.basename(save_path) + f'_{epoch+1}.pth'))
            logging.info(f"Model saved to {os.path.join(save_path, os.path.basename(save_path) + f'_{epoch+1}.pth')}")
    
    show_image(model, valid_loader)
    torch.save(model.state_dict(), os.path.join(save_path, os.path.basename(save_path) + f"_{NUM_EPOCHS}.pth"))
    logging.info(f"Model saved to {os.path.join(save_path, os.path.basename(save_path) + f'_{NUM_EPOCHS}.pth')}")


def show_image(model, valid_loader):
    model.eval()

    with torch.no_grad():
        data = next(iter(valid_loader))[0].to(DEVICE)
        decoded = model(data)['x_hat']
        residual = data - decoded

        num_images = min(8, data.size(0))
        fig, axs = plt.subplots(3, num_images, figsize=(num_images * 2, 6))

        for i in range(num_images):
            # Original
            axs[0, i].imshow(data[i].cpu().permute(1, 2, 0))
            axs[0, i].set_title("Original")
            axs[0, i].axis("off")

            # Reconstructed
            axs[1, i].imshow(decoded[i].cpu().permute(1, 2, 0))
            axs[1, i].set_title("Reconstructed")
            axs[1, i].axis("off")

            # Residual
            res_img = (residual[i].cpu().permute(1, 2, 0) + 0.5).clamp(0, 1)
            axs[2, i].imshow(res_img)
            axs[2, i].set_title("Residual")
            axs[2, i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset_path = './dataset/dogs'
    train_dataset = CustomDataset(root_dir=os.path.join(dataset_path, 'train'), transform=transform)
    valid_dataset = CustomDataset(root_dir=os.path.join(dataset_path, 'valid'), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    

    parser = argparse.ArgumentParser(description='Train an Autoencoder.')
    parser.add_argument('--model-name', type=str, default=f"AE_S{IMAGE_SIZE}_B{BOTTLENECK_FILTERS}_BS{BATCH_SIZE}", help='Name of the model to save')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=NUM_EPOCHS, help='Number of epochs for training')
    parser.add_argument('--save-interval', type=int, default=SAVE_INTER, help='Interval for saving the model')
    args = parser.parse_args()

    if not os.path.exists("train_models"):
        os.makedirs("train_models")
    save_path = os.path.join("train_models", args.model_name)
    if os.path.exists(save_path):
        logging.warning(f"Model {args.model_name} already exists. Overwriting.")
    else:
        os.mkdir(save_path)

    if args.batch_size:
        BATCH_SIZE = args.batch_size
    if args.num_epochs:
        NUM_EPOCHS = args.num_epochs
    if args.save_interval:
        SAVE_INTER = args.save_interval
    
    start = time.time()
    logging.info("Training started")
    logging.info(f"Model name: {args.model_name}")
    logging.info(f"Batch size: {BATCH_SIZE}")
    logging.info(f"Learning rate: {LR}")
    logging.info(f"Epochs: {NUM_EPOCHS}")
    logging.info(f"Image size: {IMAGE_SIZE}")
    logging.info(f"Device: {DEVICE}")

    train(train_loader, valid_loader, save_path, usesWandb=False)

    logging.info(f"Training completed in {(time.time() - start)//60:.2f} min")
    
    # test_image_path = './dataset/dogs/valid/n02085936_1390.JPEG'
    # test_model(model_path, test_image_path)