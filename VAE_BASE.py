import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Data Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])

train_data = MNIST(root='./data', train=True, transform=transform, download=True)
test_data = MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

class VAE(nn.Module):
    def __init__(self, input_size, hidden_dim=256, latent_dim=128):
        super(VAE, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size*input_size, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_size*input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(z))  

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size*self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 28*28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(input_size=28).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
num_epochs = 15
log_interval = 200

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
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
    for data, _ in test_loader:
        data = data.to(device)
        recon_batch, mu, logvar = model(data)
        test_loss += vae_loss(recon_batch, data, mu, logvar).item()
        print('test', recon_batch.shape)

    test_loss /= len(test_loader.dataset)
    print('Test Loss: {:.6f}'.format(test_loss))

    num_images = 8
    fig, axs = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        axs[0, i].imshow(data[i][0].cpu(), cmap='gray')
        axs[0, i].set_title("Original")
        axs[0, i].axis("off")

        axs[1, i].imshow(recon_batch[i].view(28, 28).cpu(), cmap='gray')
        axs[1, i].set_title("Reconstructed")
        axs[1, i].axis("off")

    plt.show()
