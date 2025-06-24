import os
import cv2
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to your dataset
dataset_path = 'cropped_dataset'  # Folder with class subfolders
target_image_count = 200  # Target number of images per class

# Augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to load image
def load_image(path):
    return cv2.imread(path)

# Augment each class
for class_name in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_folder):
        continue

    images = os.listdir(class_folder)
    image_paths = [os.path.join(class_folder, img) for img in images]
    current_count = len(images)

    print(f"Augmenting class: {class_name} | Current: {current_count} | Target: {target_image_count}")

    image_index = current_count + 1  # Start naming new images

    while current_count < target_image_count:
        # Randomly pick an existing image
        random_image_path = random.choice(image_paths)
        img = load_image(random_image_path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)

        # Generate one augmented image
        for batch in datagen.flow(img, batch_size=1):
            new_img = batch[0].astype('uint8')
            new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(class_folder, f'{image_index}.jpg')
            cv2.imwrite(save_path, new_img)

            image_index += 1
            current_count += 1
            break  # Generate one image per loop

    print(f"Completed class: {class_name} | Final count: {current_count}")

print("✅ All classes now have 200 images each.")
