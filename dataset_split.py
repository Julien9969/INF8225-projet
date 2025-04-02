import os, tqdm, shutil, random, argparse

TRAIN_PERCENT = 0.8
TOP_FOLDERS = ['dogs', 'pizza', 'shuffle']
SEED = 41

def split_dataset(top_folder: str):

    os.makedirs(os.path.join('dataset', top_folder, 'train'), exist_ok=True)
    os.makedirs(os.path.join('dataset', top_folder, 'valid'), exist_ok=True)

    files = [file for file in os.listdir(os.path.join('dataset', top_folder)) if file.endswith('.JPEG')]
    random.shuffle(files)

    train_count = int(len(files) * TRAIN_PERCENT)
    valid_count = len(files) - train_count

    print(f"Splitting {top_folder} dataset into {train_count} train and {valid_count} valid files")
    for i in tqdm.tqdm(range(len(files)), desc=f"Moving {top_folder} files"):
        target_folder = 'train' if i < train_count else 'valid'
        shutil.move(os.path.join('dataset', top_folder, files[i]),
                    os.path.join('dataset', top_folder, target_folder, files[i]))
        
def restore_dataset(top_folder: str):
    if not os.path.exists(os.path.join('dataset', top_folder, 'train')) or \
       not os.path.exists(os.path.join('dataset', top_folder, 'valid')):
        print(f"Dataset {top_folder} already restored.")
        return
    
    print(f"Restoring {top_folder} dataset")
    for folder in ['train', 'valid']:
        files = os.listdir(os.path.join('dataset', top_folder, folder))
        for file in tqdm.tqdm(files, desc=f"Restoring {top_folder} files"):
            shutil.move(os.path.join('dataset', top_folder, folder, file),
                        os.path.join('dataset', top_folder, file))
            
    if len(os.listdir(os.path.join('dataset', top_folder, 'train'))) == 0:
        os.rmdir(os.path.join('dataset', top_folder, 'train'))
    if len(os.listdir(os.path.join('dataset', top_folder, 'valid'))) == 0:
        os.rmdir(os.path.join('dataset', top_folder, 'valid'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into train and valid folders')
    parser.add_argument('--restore', action='store_true', help='Restore the dataset to its original state')
    args = parser.parse_args()

    random.seed(SEED)
    if args.restore:
        for folder in TOP_FOLDERS:
            restore_dataset(folder)
    else:
        for folder in TOP_FOLDERS:
            split_dataset(folder)