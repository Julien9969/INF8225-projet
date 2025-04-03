# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, 0 


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 256)
        self.fc21 = nn.Linear(256, 128)
        self.fc22 = nn.Linear(256, 128)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(-1, 7 * 7 * 64)
        x = nn.functional.relu(self.fc1(x))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 7 * 7 * 64)
        self.conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        z = nn.functional.relu(self.fc1(z))
        z = nn.functional.relu(self.fc2(z))
        z = z.view(-1, 64, 7, 7)
        z = nn.functional.relu(self.conv1(z))
        z = torch.tanh(self.conv2(z))
        return z

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy_with_logits(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(train_loader, valid_loader, save_path):
    model = VAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    log_interval = 100

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % log_interval == 0:
                current_loss = train_loss / (batch_idx + 1)
                print('Epoch: {} [{}/{} ({:.0f}%)]\tTraining Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    current_loss))

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in valid_loader:
            data = data.to(DEVICE)
            recon_batch, mu, logvar = model(data)
            test_loss += vae_loss(recon_batch, data, mu, logvar).item()

        test_loss /= len(valid_loader.dataset)
        print('Test Loss: {:.6f}'.format(test_loss))

        num_images = 8
        import matplotlib.pyplot as plt
        for i in range(num_images):
            sample_data, _ = next(iter(valid_loader))
            sample_data = sample_data.to(DEVICE)
            recon_batch_sample, _, _ = model(sample_data)

            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(sample_data[i][0].cpu().detach().numpy(), cmap='gray')
            axs[0].set_title('Original Image')
            axs[1].imshow(recon_batch_sample[i][0].cpu().detach().numpy(), cmap='gray')
            axs[1].set_title('Reconstructed Image')
            plt.show()
            plt.close(fig)

    torch.save(model.state_dict(), os.path.join(save_path))
    print(f"Trained model saved to: {os.path.join(save_path)}")


def compress_image(image, model):
    """Compresses an image using the VAE encoder.

    Args:
        image (torch.Tensor): Input image tensor (shape: [1, 1, 28, 28]).
        model (VAE): Loaded VAE model.

    Returns:
        torch.Tensor: The mean of the latent distribution (compressed representation).
    """
    with torch.no_grad():
        mu, logvar = model.encoder(image.to(DEVICE))
    return mu.cpu()


def decompress_image(latent_vector, model):
    """Decompresses a latent vector using the VAE decoder.

    Args:
        latent_vector (torch.Tensor): The latent vector (mean).
        model (VAE): Loaded VAE model.

    Returns:
        torch.Tensor: The reconstructed image tensor (shape: [1, 1, 28, 28]).
    """
    with torch.no_grad():
        reconstructed_image = model.decoder(latent_vector.to(DEVICE))
    return torch.sigmoid(reconstructed_image).cpu() 

def test_model(model_path, image_path):
    loaded_model = VAE().to(DEVICE)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()
    print("Trained model loaded successfully.")

    sample_image, _ = Image.open(image_path), 0
    sample_image = sample_image[0].unsqueeze(0).to(DEVICE)

    compressed_representation = compress_image(sample_image, loaded_model)
    print("Shape of compressed representation:", compressed_representation.shape)

    decompressed_image = decompress_image(compressed_representation, loaded_model)

    original_image_np = sample_image.cpu().squeeze().numpy()
    decompressed_image_np = decompressed_image.squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original_image_np, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(decompressed_image_np, cmap='gray')
    axes[1].set_title('Decompressed Image')
    axes[1].axis('off')
    plt.show()


if __name__ == "__main__":
    dogs_data_dir = './dataset/dogs'

    train_dataset = CustomDataset(root_dir=os.path.join(dogs_data_dir, 'train'), transform=transform)
    valid_dataset = CustomDataset(root_dir=os.path.join(dogs_data_dir, 'valid'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=12, shuffle=False)
