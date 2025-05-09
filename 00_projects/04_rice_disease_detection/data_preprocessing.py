import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

input_dir = "dataset"
output_dir = "preprocessed_data"
img_size = 128

def create_directory_structure():
    for split in ['train', 'val', 'test']:
        for category in os.listdir(input_dir):
            path = os.path.join(output_dir, split, category)
            os.makedirs(path, exist_ok=True)

def split_dataset():
    for category in os.listdir(input_dir):
        images = os.listdir(os.path.join(input_dir, category))
        total = len(images)
        train_end = int(total * 0.7)
        val_end = int(total * 0.85)

        for i, img in enumerate(images):
            src = os.path.join(input_dir, category, img)
            if i < train_end:
                dst = os.path.join(output_dir, 'train', category, img)
            elif i < val_end:
                dst = os.path.join(output_dir, 'val', category, img)
            else:
                dst = os.path.join(output_dir, 'test', category, img)

            img_arr = cv2.imread(src)
            resized = cv2.resize(img_arr, (img_size, img_size))
            cv2.imwrite(dst, resized)

create_directory_structure()
split_dataset()