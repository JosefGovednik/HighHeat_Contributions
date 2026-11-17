import os
import shutil

# Paths to the datasets to merge
datasets = [
    "path/to/dataset1",  # Replace with the path to the first dataset
]

# Path to the output merged dataset
output_images = "\hhdata_v0\obj_train_data\images"
output_labels = "\hhdata_v0\obj_train_data\labels"

# Create output directories
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# Merge datasets
for dataset in datasets:
    images_folder = os.path.join(dataset, "images")
    labels_folder = os.path.join(dataset, "labels")

    # Copy images
    for file in os.listdir(images_folder):
        if file.endswith(('.jpg', '.png')):
            shutil.copy(os.path.join(images_folder, file), os.path.join(output_images, file))

    # Copy labels
    for file in os.listdir(labels_folder):
        if file.endswith('.txt'):
            shutil.copy(os.path.join(labels_folder, file), os.path.join(output_labels, file))

print("Datasets merged successfully!")