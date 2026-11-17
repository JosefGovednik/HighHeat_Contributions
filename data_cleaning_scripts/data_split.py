import os
import random
import shutil

# Paths to your dataset folders
images_folder = "hhdata_v0/obj_train_data/images"  # Replace with the path to your images folder
labels_folder = "hhdata_v0/obj_train_data/labels"  # Replace with the path to your labels folder
output_folder = "hhdata_v0/obj_train_data/output"  # Replace with the path to save the split dataset

# Train/validation split ratio
train_ratio = 0.8  # 80% for training, 20% for validation

# Create output directories
train_images = os.path.join(output_folder, "train/images")
train_labels = os.path.join(output_folder, "train/labels")
val_images = os.path.join(output_folder, "val/images")
val_labels = os.path.join(output_folder, "val/labels")

os.makedirs(train_images, exist_ok=True)
os.makedirs(train_labels, exist_ok=True)
os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png'))]

# Shuffle the files for randomness
random.shuffle(image_files)

# Split the dataset
split_index = int(len(image_files) * train_ratio)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# Copy files to train and val folders
for file in train_files:
    # Copy images
    shutil.copy(os.path.join(images_folder, file), os.path.join(train_images, file))
    # Copy corresponding labels
    label_file = file.replace('.jpg', '.txt').replace('.png', '.txt')
    shutil.copy(os.path.join(labels_folder, label_file), os.path.join(train_labels, label_file))

for file in val_files:
    # Copy images
    shutil.copy(os.path.join(images_folder, file), os.path.join(val_images, file))
    # Copy corresponding labels
    label_file = file.replace('.jpg', '.txt').replace('.png', '.txt')
    shutil.copy(os.path.join(labels_folder, label_file), os.path.join(val_labels, label_file))

print("Dataset split complete!")