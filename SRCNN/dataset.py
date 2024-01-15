from datasets import load_dataset
from PIL import Image
import os
import shutil

dataset = load_dataset("eugenesiow/Div2k", "bicubic_x2")
hr_dir = '../hr/images'
lr_dir = '../lr/images'
def copy_all_files(src_dir, dst_dir):
    for img in os.listdir(src_dir):
        shutil.copy(os.path.join(src_dir, img), dst_dir)
def data():
    print("hi")
    dataset = load_dataset("eugenesiow/Div2k", "bicubic_x2")
    print(dataset['train'])
    hr_dir = './hr/images'
    lr_dir = './lr/images'
    print(os.path.abspath(lr_dir))
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    # Assuming dataset is structured with 'train', 'validation' splits
    for split in ['train', 'validation']:
        for idx, item in enumerate(dataset[split]):
            # High-resolution images
            print(item)
            print(idx)
            print(split)
            hr_image_target = os.path.join(hr_dir, f'{split}_{idx}_hr.png')  # New path for the HR image
            hr_image_path = "/Users/haorancheng/.cache/huggingface/datasets/downloads/extracted/cec4f6d77348a284f0eef37f472f14a80a2b5a17c13ad6b954fbfe7010cee9f2/DIV2K_train_HR"
            copy_all_files(hr_image_path,hr_dir)
            #shutil.copy(hr_image_path, hr_image_target)
            
            # Low-resolution images (change 'lr' to match dataset structure if different)
            lr_image_target = os.path.join(hr_dir, f'{split}_{idx}_lr.png')  # New path for the HR image
            lr_image_path = "/Users/haorancheng/.cache/huggingface/datasets/downloads/extracted/0f843f42723f0ff0ea72fce590e78babfb0ee785dc5e7307ccb7d874aa95835f/DIV2K_train_LR_bicubic/x2"
            copy_all_files(lr_image_path,lr_dir)
            
if __name__ == "__main__":
    print("hello")
    data()
