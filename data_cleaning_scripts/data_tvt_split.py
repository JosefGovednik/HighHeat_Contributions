import os
import random
import shutil

# Paths to your dataset folders
images_folder = "hhdata_v0/obj_train_data/images"  # Replace with the path to your images folder
labels_folder = "hhdata_v0/obj_train_data/labels"  # Replace with the path to your labels folder
output_folder = "hhdata_v0/obj_train_data/output"  # Replace with the path to save the split dataset

# Train/validation/test split ratios
train_ratio = 0.8  # 80% for training
val_ratio = 0.1    # 10% for validation
test_ratio = 0.1   # 10% for testing

# Create output directories
train_images = os.path.join(output_folder, "train/images")
train_labels = os.path.join(output_folder, "train/labels")
val_images = os.path.join(output_folder, "val/images")
val_labels = os.path.join(output_folder, "val/labels")
test_images = os.path.join(output_folder, "test/images")
test_labels = os.path.join(output_folder, "test/labels")

os.makedirs(train_images, exist_ok=True)
os.makedirs(train_labels, exist_ok=True)
os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)
os.makedirs(test_images, exist_ok=True)
os.makedirs(test_labels, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png'))]

# Remove images without associated labels
valid_image_files = []
for file in image_files:
    label_file = file.replace('.jpg', '.txt').replace('.png', '.txt')
    label_path = os.path.join(labels_folder, label_file)
    if os.path.exists(label_path):
        valid_image_files.append(file)  # Keep the image if the label exists
    else:
        # Delete the image file if no label is found
        os.remove(os.path.join(images_folder, file))
        print(f"Deleted image without label: {file}")

# Shuffle the valid files for randomness
random.shuffle(valid_image_files)

# Split the dataset
train_split = int(len(valid_image_files) * train_ratio)
val_split = int(len(valid_image_files) * (train_ratio + val_ratio))

train_files = valid_image_files[:train_split]
val_files = valid_image_files[train_split:val_split]
test_files = valid_image_files[val_split:]

# Copy files to train, val, and test folders
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

for file in test_files:
    # Copy images
    shutil.copy(os.path.join(images_folder, file), os.path.join(test_images, file))
    # Copy corresponding labels
    label_file = file.replace('.jpg', '.txt').replace('.png', '.txt')
    shutil.copy(os.path.join(labels_folder, label_file), os.path.join(test_labels, label_file))

print("Dataset split into train, validation, and test sets complete!")