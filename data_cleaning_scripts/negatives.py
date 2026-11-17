import os
import random
import shutil

# Paths to the excess folder and dataset folders
excess_images_folder = "hhdata_v0/obj_train_data/excess/images"  # Path to excess images
excess_labels_folder = "hhdata_v0/obj_train_data/excess/labels"  # Path to excess labels
train_images_folder = "hhdata_v0/obj_train_data/output/train/images"
train_labels_folder = "hhdata_v0/obj_train_data/output/train/labels"
val_images_folder = "hhdata_v0/obj_train_data/output/val/images"
val_labels_folder = "hhdata_v0/obj_train_data/output/val/labels"
test_images_folder = "hhdata_v0/obj_train_data/output/test/images"
test_labels_folder = "hhdata_v0/obj_train_data/output/test/labels"

# Number of excess frames to add
num_frames_to_add = 200

# Get all excess image files
excess_files = [f for f in os.listdir(excess_images_folder) if f.endswith(('.jpg', '.png'))]

# Randomly select 200 excess files
selected_files = random.sample(excess_files, min(num_frames_to_add, len(excess_files)))

# Split the selected files into train, val, and test sets
train_split = int(len(selected_files) * 0.8)
val_split = int(len(selected_files) * 0.9)

train_files = selected_files[:train_split]
val_files = selected_files[train_split:val_split]
test_files = selected_files[val_split:]

# Function to move files to the respective dataset folders
def move_files(files, src_images, src_labels, dest_images, dest_labels):
    for file in files:
        # Move the image file
        shutil.copy(os.path.join(src_images, file), os.path.join(dest_images, file))
        # Move the corresponding label file
        label_file = file.replace('.jpg', '.txt').replace('.png', '.txt')
        shutil.copy(os.path.join(src_labels, label_file), os.path.join(dest_labels, label_file))

# Add files to train, val, and test folders
move_files(train_files, excess_images_folder, excess_labels_folder, train_images_folder, train_labels_folder)
move_files(val_files, excess_images_folder, excess_labels_folder, val_images_folder, val_labels_folder)
move_files(test_files, excess_images_folder, excess_labels_folder, test_images_folder, test_labels_folder)

print(f"Added {len(train_files)} frames to train, {len(val_files)} to val, and {len(test_files)} to test.")