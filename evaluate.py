import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import logging, time, tqdm, sys
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from rich import table
import rich


from AE import Autoencoder, CustomDataset, transform

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s - %(asctime)s', datefmt='%H:%M:%S')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SHOW = False

def load_model(model_path: str):
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(DEVICE)
    return model

def evaluate(model, dataloader):
    PSRN = []
    SSIM = []
    compression_ratios = []

    with torch.no_grad():
        with tqdm.tqdm(dataloader, desc=f"Calcules métriques", unit="image", position=0, leave=False, ncols=100) as t_loader:

            for inputs, _ in t_loader:
                inputs = inputs.to(DEVICE)
                encode = model.encoder(inputs)
                decode = model.decoder(encode)

                psnr_value = psnr(inputs[0].cpu().numpy(), decode[0].cpu().numpy(), data_range=1)
                ssim_value = ssim(inputs[0].cpu().numpy(), decode[0].cpu().numpy(), multichannel=True, channel_axis=0, data_range=1)

                PSRN.append(psnr_value)
                SSIM.append(ssim_value)

                original_size = inputs.numel() * inputs.element_size()
                compressed_size = encode.numel() * encode.element_size()
                compression_ratio = original_size / compressed_size
                compression_ratios.append(compression_ratio)

                    
                if SHOW:
                    with open("encode", "wb") as f:
                        np.save(f, encode[0].cpu().numpy())
                    
                    with open("decode", "wb") as f:
                        np.save(f, decode[0].cpu().numpy())
                    
                    with open("inputs", "wb") as f:
                        np.save(f, inputs[0].cpu().numpy())

                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

                    original = inputs[0].cpu().permute(1, 2, 0)
                    reconstructed = decode[0].cpu().permute(1, 2, 0)
                    residual = (inputs[0] - decode[0]).cpu().permute(1, 2, 0).clamp(0, 1)

                    axs[0].imshow(original)
                    axs[0].set_title("Original")
                    axs[0].axis("off")

                    axs[1].imshow(reconstructed)
                    axs[1].set_title("Reconstructed")
                    axs[1].axis("off")

                    axs[2].imshow(residual)
                    axs[2].set_title("Residual")
                    axs[2].axis("off")

                    plt.show()

                logging.debug(f"PSNR: {psnr_value}, SSIM: {ssim_value}, Compression Ratio: {compression_ratio}")

    return PSRN, SSIM, compression_ratios


if __name__ == "__main__":
    model_path = "train_models/AE_128_250.pth"
    model = load_model(model_path)

    validate_dir = "dataset/dogs/valid"
    validate_dataset = CustomDataset(validate_dir, transform=transform)

    validate_dataloader = DataLoader(validate_dataset, batch_size=1, shuffle=False)
    
    PSNR, SSIM, compression_ratios = evaluate(model, validate_dataloader)

    table = table.Table(title="Metrics", show_lines=True)
    table.add_column("Metric", justify="center")
    table.add_column("Mean", justify="center")
    table.add_column("Std", justify="center")
    table.add_column("Min", justify="center")
    table.add_column("Max", justify="center")

    table.add_row("PSNR", f"{np.mean(PSNR):.2f}", f"{np.std(PSNR):.2f}", f"{np.min(PSNR):.2f}", f"{np.max(PSNR):.2f}")
    table.add_row("SSIM", f"{np.mean(SSIM):.2f}", f"{np.std(SSIM):.2f}", f"{np.min(SSIM):.2f}", f"{np.max(SSIM):.2f}")
    table.add_row("Compression Ratio", f"{np.mean(compression_ratios):.2f}", f"{np.std(compression_ratios):.2f}", f"{np.min(compression_ratios):.2f}", f"{np.max(compression_ratios):.2f}")

    rich.print(table)