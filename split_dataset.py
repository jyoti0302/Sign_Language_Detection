import os
import shutil
import random

# Paths
dataset_path = 'cropped_dataset'  # Folder where your balanced dataset is stored (classes inside)
output_path = 'final_dataset'  # Folder where train and test folders will be created
train_ratio = 0.8  # 80% training, 20% testing

# Create train and test folders
train_path = os.path.join(output_path, 'train')
test_path = os.path.join(output_path, 'test')
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Split each class
for class_name in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_folder):
        continue

    images = os.listdir(class_folder)
    random.shuffle(images)

    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Create class folders in train and test
    os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_path, class_name), exist_ok=True)

    # Copy training images
    for img in train_images:
        src = os.path.join(class_folder, img)
        dst = os.path.join(train_path, class_name, img)
        shutil.copy(src, dst)

    # Copy testing images
    for img in test_images:
        src = os.path.join(class_folder, img)
        dst = os.path.join(test_path, class_name, img)
        shutil.copy(src, dst)

    print(f"Class '{class_name}' split: {len(train_images)} train, {len(test_images)} test")

print("✅ Dataset successfully split into train and test sets.")
