import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s - %(asctime)s - %(levelname)s')

IMAGE_SIZE = 128

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NUM_EPOCHS = 500
# BATCH_SIZE = 64
LR = 0.001

NUM_EPOCHS = 1000
BATCH_SIZE = 128

HIDDEN_DIM = 256
LATENT_DIM = 1024
# HIDDEN_DIM = 514
# LATENT_DIM = 256
WEIGHT_DECAY = 1e-5

# CONV_N_FILTERS_1 = 6 # multiple de 3 idealement?
# CONV_N_FILTERS_2 = 16 #
NUM_CHANNELS = 3
CONV_FILTERS_1 = 16 
CONV_FILTERS_2 = 32
CONV_FILTERS_3 = 64
CONV_FILTERS_4 = 128 



print(f"Using device: {DEVICE}")

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


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, CONV_FILTERS_1, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(CONV_FILTERS_1, CONV_FILTERS_2, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(CONV_FILTERS_2, CONV_FILTERS_3, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(CONV_FILTERS_3, CONV_FILTERS_4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(CONV_FILTERS_4, CONV_FILTERS_3, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(CONV_FILTERS_3, CONV_FILTERS_2, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(CONV_FILTERS_2, CONV_FILTERS_1, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(CONV_FILTERS_1, NUM_CHANNELS, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(train_loader, valid_loader, save_path):
    # model = VAE(input_size=IMAGE_SIZE).to(DEVICE)
    model = Autoencoder().to(DEVICE)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for data, _ in train_loader:
            # print(data.shape) # BATCH, COLOR, SIZE, SIZE
            data = data.to(DEVICE)
            optimizer.zero_grad()
            recon_batch = model(data)
            loss = mse_loss(recon_batch, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader.dataset)
        logging.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}")
        if (epoch + 1) % 50 == 0:
            evaluate(model, valid_loader, mse_loss)
            
    
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model saved to {save_path}")

def evaluate(model, valid_loader, mse_loss):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in valid_loader:
            data = data.to(DEVICE)
            recon_batch = model(data)
            test_loss += mse_loss(recon_batch, data).item()

        test_loss /= len(valid_loader.dataset)
        logging.info('Test Loss: {:.6f}'.format(test_loss))

        # On reprend data du debut prcq les batch sont trop petits à la fin
        data = next(iter(valid_loader))[0].to(DEVICE)
        recon_batch = model(data)
        num_images = 8
        fig, axs = plt.subplots(2, 8, figsize=(12, 3))

        # print("===========", data.shape)
        # print("==--------=========", recon_batch.shape)
        # print("===========++", data[0, :].shape)
        for i in range(num_images):
            axs[0, i].imshow(data[i, :].cpu().squeeze().permute(1, 2, 0))
            axs[0, i].set_title("Original")
            axs[0, i].axis("off")

            axs[1, i].imshow(recon_batch[i, :].cpu().permute(1, 2, 0))
            axs[1, i].set_title("Reconstructed")
            axs[1, i].axis("off")

    plt.show()

# def test_model(model_path, image_path):
#     model = VAE_Class(256).to(DEVICE)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()

#     image = Image.open(image_path).convert('L')
#     image = transform(image).unsqueeze(0).to(DEVICE)
    
#     with torch.no_grad():
#         mu, logvar = model.encode(image)
#         latent_vector = model.reparameterize(mu, logvar)
#         recon_image = model.decode(latent_vector).cpu().squeeze().numpy()
    
#     fig, axes = plt.subplots(1, 2, figsize=(8, 4))
#     axes[0].imshow(image.cpu().numpy().squeeze().permute(1, 2, 0))
#     axes[0].set_title('Original')
#     axes[1].imshow(recon_image.permute(1, 2, 0))
#     axes[1].set_title('Reconstructed')
#     plt.show()

if __name__ == "__main__":
    dataset_path = './dataset/dogs'
    train_dataset = CustomDataset(root_dir=os.path.join(dataset_path, 'train'), transform=transform)
    valid_dataset = CustomDataset(root_dir=os.path.join(dataset_path, 'valid'), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model_path = "vae_model.pth"
    train(train_loader, valid_loader, model_path)
    
    # test_image_path = './dataset/dogs/valid/n02085936_1390.JPEG'
    # test_model(model_path, test_image_path)