import os, shutil, random


IMAGES_FOLDER="G:/IMAGE-NET/ILSVRC/Data/CLS-LOC/train"
CATEGORY_MAPPING="G:/IMAGE-NET/ILSVRC/Data/CLS-LOC/LOC_synset_mapping.txt"
SEED=42


def get_images(category: str, out_dir: str, limit: int=-1, shuffle: bool=False):
    """
    Get images from the ImageNet dataset efficiently.
    """
    folders = os.path.join(IMAGES_FOLDER, category)
    images = [entry.name for entry in os.scandir(folders) if entry.is_file() and entry.name.endswith(".JPEG")]
    if not images:  
        return

    if shuffle and limit > 0:
        images = random.sample(images, min(limit, len(images)))  
    elif limit > 0:
        images = images[:limit]  

    os.makedirs(out_dir, exist_ok=True)

    for image in images:
        image_path = os.path.join(folders, image)
        out_path = os.path.join(out_dir, image)
        shutil.copy(image_path, out_path)


def parse_category_mapping(mapping_file: str):
    """
    Parse the category mapping file.
    ex: n01440764 tench, Tinca tinca
    """
    mapping = {}
    with open(mapping_file, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" ", 1) 
            if len(parts) == 2:
                key, values = parts
                mapping[key] = values
    return mapping

def filter_categories(categories: list, value_contains: str):
    return [ dog for dog in categories.keys() if value_contains in categories[dog] ]

if __name__ == "__main__":
    random.seed(SEED)

    out_dir = os.path.join(os.curdir, "dataset")
    os.makedirs(out_dir, exist_ok=True)

    category_mapping = parse_category_mapping(CATEGORY_MAPPING)
    
    dog_categories = filter_categories(category_mapping, "dog")
    pizza_categories = filter_categories(category_mapping, "pizza")
    
    print(f"Dog categories: {len(dog_categories)}")
    print(f"Pizza categories: {len(pizza_categories)}")

    for category in dog_categories:
        print(f"Dog category: {category} - {category_mapping[category]}")
        get_images(category, os.path.join(out_dir, 'dogs'), limit=82, shuffle=True)

    # # for category in pizza_categories:
    # #     get_images(category, os.path.join(out_dir, 'pizza'))

    # for category in category_mapping.keys():
    #     print(f"Shuffle Category: {category} - {category_mapping[category]}")
        
    #     # limit_value = 1 if random.random() > 0.3 else 2 # Hack to have ~1300 images
    #     limit_value = 0 if random.random() > 0.3 else 1 # Hack to have ~1300 images

    #     if limit_value == 0:
    #         continue
    #     get_images(category, os.path.join(out_dir, 'shuffle'), limit=limit_value, shuffle=True)
