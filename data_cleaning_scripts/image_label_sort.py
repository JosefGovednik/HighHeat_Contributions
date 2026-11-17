import os
import shutil

# Path to the dataset folder containing all files
dataset_folder = "hhdata_v1/obj_train_data"  # Replace with the path to your dataset folder

# Paths to the target folders
images_folder = os.path.join(dataset_folder, "images")
labels_folder = os.path.join(dataset_folder, "labels")

# Create target directories if they don't exist
os.makedirs(images_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)

# Iterate through all files in the dataset folder
for file in os.listdir(dataset_folder):
    file_path = os.path.join(dataset_folder, file)

    # Skip directories
    if os.path.isdir(file_path):
        continue

    # Move .png files to the images folder
    if file.endswith('.png'):
        shutil.move(file_path, os.path.join(images_folder, file))

    # Move .txt files to the labels folder
    elif file.endswith('.txt'):
        shutil.move(file_path, os.path.join(labels_folder, file))

print("Files organized into 'images' and 'labels' folders!")