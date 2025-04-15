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
import rich, os
from compressai.entropy_models import EntropyBottleneck

from AE import Autoencoder, CustomDataset, transform
from io import BytesIO
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s - %(asctime)s', datefmt='%H:%M:%S')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SHOW = False
TEST_SAVE = False

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

def input_vs_encoded(original_image, encoded):
    buffer = BytesIO()
    original_image.save(buffer, format="JPEG", quality=100)
    ori_jpeg_size = buffer.tell()

    encoded_buffer = BytesIO()
    encoded_buffer.write(encoded.cpu().numpy())
    encoded_size = encoded_buffer.tell()
    encoded_buffer.close()

    return ori_jpeg_size, encoded_size, ori_jpeg_size / encoded_size


def jpeg_compression(original_image, compression_ratio):
    buffer = BytesIO()
    quality = int(np.clip(95 - (compression_ratio - 1) * 10, 5, 95))

    original_image.save(buffer, format="JPEG", quality=quality)
    comp_jpeg_size = buffer.tell()

    comp_jpeg_image = Image.open(buffer)
    comp_jpeg_image.load()
    buffer.close()

    return comp_jpeg_size, comp_jpeg_image

def convert_pil_to_tensor(image: Image.Image, device=None):
    """
    Convert a PIL image to a normalized torch tensor, ensuring RGB format.
    """
    tensor = transforms.ToTensor()(image.convert("RGB")).unsqueeze(0)
    return tensor.to(device) if device else tensor

def evaluate(model, dataloader):
    PSNR = []
    SSIM = []
    compression_ratios = []
    jpeg_PSNR = []
    jpeg_SSIM = []
    jpeg_compression_ratios = []
    entropy_bottleneck = EntropyBottleneck(64).to(DEVICE).eval()

    with torch.no_grad():        
        with tqdm.tqdm(dataloader, desc=f"Calcules métriques", unit="image", position=0, leave=False, ncols=100) as t_loader:
            for inputs, _ in t_loader:
                inputs = inputs.to(DEVICE)
                encoded = model.encoder(inputs)
                encoded, likelihoods = entropy_bottleneck(encoded)
                decoded = model.decoder(encoded)

                psnr_value, ssim_value = calculate_metrics(inputs[0], decoded[0])
                PSNR.append(psnr_value)
                SSIM.append(ssim_value)

                original_image = transforms.ToPILImage()(inputs[0].cpu())
                
                input_jpeg_size, encoded_size, compression_ratio = input_vs_encoded(original_image, encoded)
                compression_ratios.append(compression_ratio)

                comp_jpeg_size, comp_jpeg_image = jpeg_compression(original_image, compression_ratio)
                jpeg_ratio = input_jpeg_size / comp_jpeg_size
                jpeg_compression_ratios.append(jpeg_ratio)
                
                jpeg_tensor = convert_pil_to_tensor(comp_jpeg_image, device=DEVICE)
                jpeg_psnr_value, jpeg_ssim_value = calculate_metrics(inputs[0], jpeg_tensor[0])
                jpeg_PSNR.append(jpeg_psnr_value)
                jpeg_SSIM.append(jpeg_ssim_value)

                logging.debug(f"JPEG PSNR: {jpeg_psnr_value}, JPEG SSIM: {jpeg_ssim_value}, JPEG Compression Ratio: {jpeg_compression_ratios[-1]}")
                logging.debug(f"PSNR: {psnr_value}, SSIM: {ssim_value}, Compression Ratio: {encoded_size}")

                if TEST_SAVE:
                    if not os.path.exists("results"):
                        os.makedirs("results")
                    original_image.save(f"results/original.jpg")
                    comp_jpeg_image.save(f"results/jpeg.jpg")
                    with open("results/encoded", "wb") as f:
                        f.write(encoded.cpu().numpy())
                if SHOW:
                    visualize_results(inputs[0], decoded[0], jpeg_tensor[0], comp_jpeg_size//1000, input_jpeg_size//1000, encoded_size//1000) 

    return PSNR, SSIM, compression_ratios, jpeg_PSNR, jpeg_SSIM, jpeg_compression_ratios

import matplotlib.pyplot as plt

def visualize_results(original, reconstructed, jpeg, comp_jpeg_size, input_jpeg_size, encoded_size):
    original_np = original.cpu().permute(1, 2, 0).clamp(0, 1)
    reconstructed_np = reconstructed.cpu().permute(1, 2, 0).clamp(0, 1)
    residual_np = (original - reconstructed).cpu().permute(1, 2, 0).clamp(0, 1)
    jpeg_np = jpeg.cpu().permute(1, 2, 0).clamp(0, 1)

    titles = [
        f"Original\n({input_jpeg_size:.1f} kB)",
        f"Encoded \n({encoded_size:.1f} kB)",
        f"",
        f"JPEG\n({comp_jpeg_size:.1f} kB)"
    ]

    fig, axs = plt.subplots(1, 4, figsize=(14, 6))
    images = [original_np, reconstructed_np, residual_np, jpeg_np]

    for ax, img, title in zip(axs, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model_path = sys.argv[1] 
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