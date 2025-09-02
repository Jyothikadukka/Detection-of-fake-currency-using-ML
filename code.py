import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

# Paths (Use absolute paths for both input and output folders)
input_folder =   "real"# Update this path
output_folder = "fake" # Update this path

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

def add_blur(img):
    return cv2.GaussianBlur(img, (9, 9), 0)

def add_noise(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def color_shift(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    shift = random.randint(-10, 10)
    h_channel = hsv[..., 0].astype(np.int16)  # Convert to int16 to avoid overflow
    h_channel = (h_channel + shift) % 180
    hsv[..., 0] = h_channel.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def skew_image(img):
    rows, cols, ch = img.shape
    pts1 = np.float32([[0, 0], [cols, 0], [0, rows]])
    pts2 = np.float32([[0, 0], [cols, random.randint(-30, 30)], [random.randint(-30, 30), rows]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (cols, rows))

def add_fake_stamp(img):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()
    draw.text((10, 10), "FAKE", fill=(255, 0, 0), font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Transformation pipeline
def apply_transformations(img):
    funcs = [add_blur, add_noise, color_shift, skew_image, add_fake_stamp]
    random.shuffle(funcs)
    for func in funcs[:3]:  # Apply 3 random transformations
        img = func(img)
    return img

# Print current working directory to ensure we are in the correct folder
print("Current working directory:", os.getcwd())

# Process all real images
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        fake_img = apply_transformations(img)
        out_path = os.path.join(output_folder, "fake_" + filename)
        cv2.imwrite(out_path, fake_img)

print("✅ Fake note dataset generated successfully!")
