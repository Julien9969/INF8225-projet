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
from rich.table import Table
import rich


from AE import Autoencoder, CustomDataset, transform
from io import BytesIO
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s - %(asctime)s', datefmt='%H:%M:%S')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SHOW = False

def load_model(model_path: str):
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(DEVICE)
    return model

def calculate_metrics(original, decoded):
    original_np = original.permute(1, 2, 0).cpu().numpy()
    decoded_np = decoded.permute(1, 2, 0).cpu().numpy()
    psnr_value = psnr(original_np, decoded_np, data_range=1)
    ssim_value = ssim(original_np, decoded_np, channel_axis=-1, data_range=1)
    return psnr_value, ssim_value

def calculate_compression_ratio(original, encoded):
    """Calculate the compression ratio."""
    original_size = original.numel() * original.element_size()
    compressed_size = encoded.numel() * encoded.element_size()
    return original_size / compressed_size

def calculate_jpeg_compression_ratio(original_tensor, jpeg_size_bytes):
    original_size = original_tensor.numel() * original_tensor.element_size()
    return original_size / jpeg_size_bytes

def jpeg_compression(original_image, compression_ratio):
    """Compress the original image using JPEG with a similar compression ratio."""
    buffer = BytesIO()
    quality = max(1, min(95, int(100 / compression_ratio)))  # Estimate JPEG quality
    original_image.save(buffer, format="JPEG", quality=quality)
    jpeg_size = buffer.tell()
    return jpeg_size, Image.open(buffer)

def evaluate(model, dataloader):
    PSNR = []
    SSIM = []
    compression_ratios = []
    jpeg_PSNR = []
    jpeg_SSIM = []
    jpeg_compression_ratios = []

    with torch.no_grad():        
        with tqdm.tqdm(dataloader, desc=f"Calcules métriques", unit="image", position=0, leave=False, ncols=100) as t_loader:
            for inputs, _ in t_loader:
                inputs = inputs.to(DEVICE)
                encoded = model.encoder(inputs)
                decoded = model.decoder(encoded)

                psnr_value, ssim_value = calculate_metrics(inputs[0], decoded[0])
                PSNR.append(psnr_value)
                SSIM.append(ssim_value)

                compression_ratio = calculate_compression_ratio(inputs, encoded)
                compression_ratios.append(compression_ratio)

                original_image = transforms.ToPILImage()(inputs[0].cpu())
                jpeg_size, jpeg_image = jpeg_compression(original_image, compression_ratio)
                jpeg_tensor = transforms.ToTensor()(jpeg_image).unsqueeze(0).to(DEVICE)

                jpeg_psnr_value, jpeg_ssim_value = calculate_metrics(inputs[0], jpeg_tensor[0])
                jpeg_PSNR.append(jpeg_psnr_value)
                jpeg_SSIM.append(jpeg_ssim_value)
                jpeg_ratio = calculate_jpeg_compression_ratio(inputs, jpeg_size)
                jpeg_compression_ratios.append(jpeg_ratio)


                logging.debug(f"JPEG PSNR: {jpeg_psnr_value}, JPEG SSIM: {jpeg_ssim_value}, JPEG Compression Ratio: {jpeg_compression_ratios[-1]}")
                logging.debug(f"PSNR: {psnr_value}, SSIM: {ssim_value}, Compression Ratio: {compression_ratio}")

                if SHOW:
                    visualize_results(inputs[0], decoded[0])

    return PSNR, SSIM, compression_ratios, jpeg_PSNR, jpeg_SSIM, jpeg_compression_ratios

def visualize_results(original, reconstructed):
    """Visualize the original, reconstructed, and residual images."""
    residual = (original - reconstructed).cpu().permute(1, 2, 0).clamp(0, 1)
    original = original.cpu().permute(1, 2, 0)
    reconstructed = reconstructed.cpu().permute(1, 2, 0)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
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

if __name__ == "__main__":
    model_path = "train_models/AE_128_250.pth"
    model = load_model(model_path)

    validate_dir = "dataset/dogs/valid"
    validate_dataset = CustomDataset(validate_dir, transform=transform)
    validate_dataloader = DataLoader(validate_dataset, batch_size=1, shuffle=False)

    PSNR, SSIM, compression_ratios, jpeg_PSNR, jpeg_SSIM, jpeg_compression_ratios = evaluate(model, validate_dataloader)

    # Table for Autoencoder Metrics
    table_ae = Table(title="Autoencoder Metrics", show_lines=True)
    table_ae.add_column("Metric", justify="center")
    table_ae.add_column("Mean", justify="center")
    table_ae.add_column("Std", justify="center")
    table_ae.add_column("Min", justify="center")
    table_ae.add_column("Max", justify="center")

    table_ae.add_row("PSNR", f"{np.mean(PSNR):.2f}", f"{np.std(PSNR):.2f}", f"{np.min(PSNR):.2f}", f"{np.max(PSNR):.2f}")
    table_ae.add_row("SSIM", f"{np.mean(SSIM):.2f}", f"{np.std(SSIM):.2f}", f"{np.min(SSIM):.2f}", f"{np.max(SSIM):.2f}")
    table_ae.add_row("Compression Ratio", f"{np.mean(compression_ratios):.2f}", f"{np.std(compression_ratios):.2f}", f"{np.min(compression_ratios):.2f}", f"{np.max(compression_ratios):.2f}")

    # Table for JPEG Metrics
    table_jpeg = Table(title="JPEG Metrics", show_lines=True)
    table_jpeg.add_column("Metric", justify="center")
    table_jpeg.add_column("Mean", justify="center")
    table_jpeg.add_column("Std", justify="center")
    table_jpeg.add_column("Min", justify="center")
    table_jpeg.add_column("Max", justify="center")

    table_jpeg.add_row("PSNR", f"{np.mean(jpeg_PSNR):.2f}", f"{np.std(jpeg_PSNR):.2f}", f"{np.min(jpeg_PSNR):.2f}", f"{np.max(jpeg_PSNR):.2f}")
    table_jpeg.add_row("SSIM", f"{np.mean(jpeg_SSIM):.2f}", f"{np.std(jpeg_SSIM):.2f}", f"{np.min(jpeg_SSIM):.2f}", f"{np.max(jpeg_SSIM):.2f}")
    table_jpeg.add_row("Compression Ratio", f"{np.mean(jpeg_compression_ratios):.2f}", f"{np.std(jpeg_compression_ratios):.2f}", f"{np.min(jpeg_compression_ratios):.2f}", f"{np.max(jpeg_compression_ratios):.2f}")

    rich.print(table_ae)
    rich.print(table_jpeg)