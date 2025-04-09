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

IMAGE_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NUM_EPOCHS = 500
# BATCH_SIZE = 64
LR = 0.001

NUM_EPOCHS = 150
BATCH_SIZE = 55

HIDDEN_DIM = 256
LATENT_DIM = 128
# HIDDEN_DIM = 514
# LATENT_DIM = 256


# CONV_N_FILTERS_1 = 6 # multiple de 3 idealement?
# CONV_N_FILTERS_2 = 16 #
CONV_N_FILTERS_1 = 6 # multiple de 3 idealement?
CONV_N_FILTERS_2 = 16 #


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

class Encoder(nn.Module):
    def __init__(self, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, CONV_N_FILTERS_1, kernel_size=3, stride=2, padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(CONV_N_FILTERS_1, CONV_N_FILTERS_2, kernel_size=3, stride=2, padding=1)
        # self.conv2 = nn.Conv3d(32-2, 64-1, kernel_size=3, stride=2, padding=1, groups=3)
        self.fc1 = nn.Linear((IMAGE_SIZE//2)//2*(IMAGE_SIZE//2)//2 * (CONV_N_FILTERS_2), hidden_dim)
        # self.fc1 = nn.Linear(IMAGE_SIZE * IMAGE_SIZE * (64-1), hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x = self.pool(torch.relu(self.conv1(x)))
        # x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1) #.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc_mu(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, (IMAGE_SIZE//2)//2*(IMAGE_SIZE//2)//2 * (CONV_N_FILTERS_2))
        # self.conv1 = nn.ConvTranspose3d(64-1, 32-2, kernel_size=3, stride=1, padding=1, groups=3)
        # self.conv2 = nn.ConvTranspose3d(32-2, 1+2, kernel_size=3, stride=1, padding=1, groups=3)
        self.conv1 = nn.ConvTranspose2d(CONV_N_FILTERS_2, CONV_N_FILTERS_1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(CONV_N_FILTERS_1, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = z.view(-1, CONV_N_FILTERS_2, ((IMAGE_SIZE//2)//2), ((IMAGE_SIZE//2)//2))
        z = torch.relu(self.conv1(z))
        out = torch.sigmoid(self.conv2(z))
        # print("===========", out.shape)
        return out

class ConvVAE(nn.Module):
    def __init__(self, input_size, hidden_dim=256, latent_dim=128):
        super(ConvVAE, self).__init__()
        self.encoder = Encoder(hidden_dim, latent_dim)
        self.decoder = Decoder(hidden_dim, latent_dim)
        self.input_size = input_size
        # Init xavier:
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data, gain=1.0)

 
    def encode(self, x):
        return self.encoder(x)#.view(-1, 3, 1, self.input_size, self.input_size))
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# class VAE(nn.Module):
#     def __init__(self, input_size, hidden_dim=256, latent_dim=128):
#         super(VAE, self).__init__()
#         self.input_size = input_size

#         self.fc1 = nn.Linear(input_size*input_size, hidden_dim)
#         self.fc21 = nn.Linear(hidden_dim, latent_dim)
#         self.fc22 = nn.Linear(hidden_dim, latent_dim)

#         self.fc3 = nn.Linear(latent_dim, hidden_dim)
#         self.fc4 = nn.Linear(hidden_dim, input_size*input_size)

#     def encode(self, x):
#         h1 = F.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         z = F.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(z))  

#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, self.input_size*self.input_size))
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar


VAE_Class = ConvVAE



def vae_loss(recon_x, x, mu, logvar):
    # print("---", IMAGE_SIZE)
    # print("===== recon_x", recon_x.shape)
    # print("===== x", x.shape)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 3, IMAGE_SIZE, IMAGE_SIZE), reduction='sum')
    # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = 0
    return BCE + KLD

def measure_compression(model, data_loader):
    data_point = data_loader.dataset[0][0].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        print(f"DATA SIZE: {data_point.shape}")
        encoded, _ = model.encoder(data_point)
        print(f"ENCODED SIZE: {encoded.shape}")
        compression_ratio = data_point.numel() / encoded.numel()
        print(f"Facteur de compression: {compression_ratio:.2f}")

def train(train_loader, valid_loader, save_path):
    # model = VAE(input_size=IMAGE_SIZE).to(DEVICE)
    model = VAE_Class(input_size=IMAGE_SIZE, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    measure_compression(model, train_loader)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for data, _ in train_loader:
            # print(data.shape) # BATCH, COLOR, SIZE, SIZE
            data = data.to(DEVICE)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_loader.dataset)
        logging.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}")
    
    torch.save(model.state_dict(), save_path)
    logging.info(f"Model saved to {save_path}")

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in valid_loader:
            data = data.to(DEVICE)
            recon_batch, mu, logvar = model(data)
            test_loss += vae_loss(recon_batch, data, mu, logvar).item()

        test_loss /= len(valid_loader.dataset)
        logging.info('Test Loss: {:.6f}'.format(test_loss))

        # On reprend data du debut prcq les batch sont trop petits à la fin
        data = next(iter(valid_loader))[0].to(DEVICE)
        recon_batch, mu, logvar = model(data)
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

def test_model(model_path, image_path):
    model = VAE_Class(256).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        mu, logvar = model.encode(image)
        latent_vector = model.reparameterize(mu, logvar)
        recon_image = model.decode(latent_vector).cpu().squeeze().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image.cpu().numpy().squeeze().permute(1, 2, 0))
    axes[0].set_title('Original')
    axes[1].imshow(recon_image.permute(1, 2, 0))
    axes[1].set_title('Reconstructed')
    plt.show()

if __name__ == "__main__":
    dataset_path = './dataset/dogs'
    train_dataset = CustomDataset(root_dir=os.path.join(dataset_path, 'train'), transform=transform)
    valid_dataset = CustomDataset(root_dir=os.path.join(dataset_path, 'valid'), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model_path = "vae_model.pth"
    train(train_loader, valid_loader, model_path)
    
    test_image_path = './dataset/dogs/valid/n02085936_1390.JPEG'
    test_model(model_path, test_image_path)