import os
import shutil

# Path to the dataset (where both images and labels are stored)
dataset_folder = "hhdata_v1/obj_train_data"  # Replace with the path to your dataset folder
excess_folder = "hhdata_v1/obj_train_data/excess"  # Replace with the path to save excess files

# Create excess directories
excess_images = os.path.join(excess_folder, "images")
excess_labels = os.path.join(excess_folder, "labels")
os.makedirs(excess_images, exist_ok=True)
os.makedirs(excess_labels, exist_ok=True)

# Find and move empty label files and their associated images
for file in os.listdir(dataset_folder):
    if file.endswith('.txt'):  # Process only label files
        label_path = os.path.join(dataset_folder, file)
        if os.path.getsize(label_path) == 0:  # Check if the label file is empty
            # Move the label file
            shutil.move(label_path, os.path.join(excess_labels, file))

            # Move the associated image file
            image_file = file.replace('.txt', '.jpg')  # Assuming images are .jpg
            image_path = os.path.join(dataset_folder, image_file)
            if os.path.exists(image_path):
                shutil.move(image_path, os.path.join(excess_images, image_file))

print("Empty frames moved to excess folder!")