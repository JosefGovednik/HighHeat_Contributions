import os
import shutil

# Paths to the dataset
images_folder = "hhdata_v1\obj_train_data\images"  # Replace with the path to your images folder
labels_folder = "hhdata_v1\obj_train_data\labels"  # Replace with the path to your labels folder
excess_folder = "hhdata_v1\obj_train_data\excess"  # Replace with the path to save excess files

# Create excess directories
excess_images = os.path.join(excess_folder, "images")
excess_labels = os.path.join(excess_folder, "labels")
os.makedirs(excess_images, exist_ok=True)
os.makedirs(excess_labels, exist_ok=True)

# Find and move empty label files and their associated images
for label_file in os.listdir(labels_folder):
    label_path = os.path.join(labels_folder, label_file)
    if os.path.getsize(label_path) == 0:  # Check if the label file is empty
        # Move the label file
        shutil.move(label_path, os.path.join(excess_labels, label_file))

        # Move the associated image file
        image_file = label_file.replace('.txt', '.jpg')  # Assuming images are .jpg
        image_path = os.path.join(images_folder, image_file)
        if os.path.exists(image_path):
            shutil.move(image_path, os.path.join(excess_images, image_file))

print("Empty frames moved to excess folder!")