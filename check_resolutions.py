import os
from PIL import Image

DATASET_DIR = "dataset"
IMAGE_EXTENSIONS = "JPEG"
DIRECTORIES = ['dogs', 'pizza', 'shuffle']


def check_resolutions(directory):
    resolutions = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(IMAGE_EXTENSIONS):
                image_path = os.path.join(root, file)
                with Image.open(image_path) as img:
                    resolution = img.size
                    if resolution not in resolutions:
                        resolutions[resolution] = 0
                    resolutions[resolution] += 1

    print("Image resolutions and their counts:")
    for resolution, count in resolutions.items():
        print(f"{resolution}: {count}")


if __name__ == "__main__":

    for directory in DIRECTORIES:
        dir_path = os.path.join(DATASET_DIR, directory)
        if os.path.exists(dir_path):
            print(f"Checking resolutions in {directory}...")
            check_resolutions(dir_path)
        else:
            print(f"Directory {directory} does not exist.")