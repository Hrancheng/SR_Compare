# bicubic_upscale.py
from PIL import Image
import os

def upscale_image(lr_image_path, hr_image_path, upscaled_dir, scale_factor=2):
    # Load the low-resolution image
    lr_image = Image.open(lr_image_path).convert("RGB")

    # Calculate the new size based on the scale factor
    new_size = (lr_image.width * scale_factor, lr_image.height * scale_factor)

    # Upscale using bicubic interpolation
    upsampled_image = lr_image.resize(new_size, Image.BICUBIC)

    # Save the upsampled image
    upscaled_image_name = os.path.basename(lr_image_path).replace("lr", "upscaled")
    upsampled_image.save(os.path.join(upscaled_dir, upscaled_image_name))

def process_dataset(lr_dir, hr_dir, upscaled_dir):
    lr_images = [img for img in os.listdir(lr_dir) if img.endswith('.png')]
    
    for img_name in lr_images:
        lr_image_path = os.path.join(lr_dir, img_name)
        hr_image_path = os.path.join(hr_dir, img_name.replace("_lr", "_hr"))
        upscale_image(lr_image_path, hr_image_path, upscaled_dir)

if __name__ == "__main__":
    lr_dir = '../lr/images'
    hr_dir = '../hr/images'
    upscaled_dir = '../upscaled_images' 
    os.makedirs(upscaled_dir, exist_ok=True)

    process_dataset(lr_dir, hr_dir, upscaled_dir)
