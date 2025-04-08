# import os, shutil, random


# IMAGES_FOLDER="G:/IMAGE-NET/ILSVRC/Data/CLS-LOC/train"
# CATEGORY_MAPPING="G:/IMAGE-NET/ILSVRC/Data/CLS-LOC/LOC_synset_mapping.txt"
# SEED=42


# def get_images(category: str, out_dir: str, limit: int=-1, shuffle: bool=False):
#     """
#     Get images from the ImageNet dataset efficiently.
#     """
#     folders = os.path.join(IMAGES_FOLDER, category)
#     images = [entry.name for entry in os.scandir(folders) if entry.is_file() and entry.name.endswith(".JPEG")]
#     if not images:  
#         return

#     if shuffle and limit > 0:
#         images = random.sample(images, min(limit, len(images)))  
#     elif limit > 0:
#         images = images[:limit]  

#     os.makedirs(out_dir, exist_ok=True)

#     for image in images:
#         image_path = os.path.join(folders, image)
#         out_path = os.path.join(out_dir, image)
#         shutil.copy(image_path, out_path)


# def parse_category_mapping(mapping_file: str):
#     """
#     Parse the category mapping file.
#     ex: n01440764 tench, Tinca tinca
#     """
#     mapping = {}
#     with open(mapping_file, "r", encoding="utf-8") as file:
#         for line in file:
#             parts = line.strip().split(" ", 1) 
#             if len(parts) == 2:
#                 key, values = parts
#                 mapping[key] = values
#     return mapping

# def filter_categories(categories: list, value_contains: str):
#     return [ dog for dog in categories.keys() if value_contains in categories[dog] ]

# if __name__ == "__main__":
#     random.seed(SEED)

#     out_dir = os.path.join(os.curdir, "dataset")
#     os.makedirs(out_dir, exist_ok=True)

#     category_mapping = parse_category_mapping(CATEGORY_MAPPING)
    
    # dog_categories = filter_categories(category_mapping, "dog")
    # dog_categories.remove("n03218198")
    # dog_categories.remove("n07697537")
    # pizza_categories = filter_categories(category_mapping, "pizza")
    
#     print(f"Dog categories: {len(dog_categories)}")
#     print(f"Pizza categories: {len(pizza_categories)}")

    # for category in dog_categories:
    #     print(f"Dog category: {category} - {category_mapping[category]}")
    #     get_images(category, os.path.join(out_dir, 'dogs'), limit=93, shuffle=True)

#     # # for category in pizza_categories:
#     # #     get_images(category, os.path.join(out_dir, 'pizza'))

#     # for category in category_mapping.keys():
#     #     print(f"Shuffle Category: {category} - {category_mapping[category]}")
        
#     #     # limit_value = 1 if random.random() > 0.3 else 2 # Hack to have ~1300 images
#     #     limit_value = 0 if random.random() > 0.3 else 1 # Hack to have ~1300 images

#     #     if limit_value == 0:
#     #         continue
#     #     get_images(category, os.path.join(out_dir, 'shuffle'), limit=limit_value, shuffle=True)

import os
import cv2
import numpy as np
from PIL import Image
from math import cos, sin, radians

def create_image_variations(input_folder, output_folder, rotation_angles=[0, 90, 180, 270], zoom_factors=[1.0, 1.2, 0.8]):
    """
    Create rotated and zoomed versions of images in a folder.
    
    Args:
        input_folder (str): Path to folder containing original images
        output_folder (str): Path to save transformed images
        rotation_angles (list): Angles to rotate images (default: [0, 90, 180, 270])
        zoom_factors (list): Zoom factors to apply (default: [1.0, 1.2, 0.8])
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each image in input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_folder, filename)
            
            try:
                # Read image with OpenCV
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                # Get image name without extension
                name, ext = os.path.splitext(filename)
                
                # Create all combinations of rotations and zooms
                for angle in rotation_angles:
                    for zoom in zoom_factors:
                        # Skip if no transformation needed
                        if angle == 0 and zoom == 1.0:
                            continue
                            
                        # Apply rotation
                        if angle != 0:
                            rotated = rotate_image(img, angle)
                        else:
                            rotated = img.copy()
                            
                        # Apply zoom
                        if zoom != 1.0:
                            zoomed = zoom_image(rotated, zoom)
                        else:
                            zoomed = rotated.copy()
                            
                        # Save transformed image
                        new_name = f"{name}_rot{angle}_zoom{zoom}{ext}"
                        output_path = os.path.join(output_folder, new_name)
                        cv2.imwrite(output_path, zoomed)
                        
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

def rotate_image(img, angle):
    """Rotate image by specified angle in degrees"""
    height, width = img.shape[:2]
    center = (width/2, height/2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding dimensions
    abs_cos = abs(rotation_matrix[0,0])
    abs_sin = abs(rotation_matrix[0,1])
    
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    
    # Adjust rotation matrix to center
    rotation_matrix[0, 2] += bound_w/2 - center[0]
    rotation_matrix[1, 2] += bound_h/2 - center[1]
    
    # Perform rotation
    rotated = cv2.warpAffine(img, rotation_matrix, (bound_w, bound_h))
    return rotated

def zoom_image(img, factor):
    """Zoom image by specified factor (1.0 = no zoom)"""
    if factor == 1.0:
        return img.copy()
        
    height, width = img.shape[:2]
    new_height, new_width = int(height * factor), int(width * factor)
    
    # Resize image
    zoomed = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Crop or pad to maintain original dimensions
    if factor > 1.0:  # Zoom in - crop center
        start_x = (new_width - width) // 2
        start_y = (new_height - height) // 2
        zoomed = zoomed[start_y:start_y+height, start_x:start_x+width]
    else:  # Zoom out - pad with black
        pad_x = (width - new_width) // 2
        pad_y = (height - new_height) // 2
        zoomed = cv2.copyMakeBorder(zoomed, pad_y, pad_y, pad_x, pad_x, 
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return zoomed

# Example usage:
create_image_variations("./dogs", "./rotated_zoomed_dogs", rotation_angles=[0, 90, 180, 270], zoom_factors=[1.0, 1.2])