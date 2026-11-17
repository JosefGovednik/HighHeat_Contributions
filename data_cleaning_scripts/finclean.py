import os
import shutil

# Paths to the images and labels folders
images_folder = "hhdata_v0/obj_train_data/images"  # Replace with the path to your images folder
labels_folder = "hhdata_v0/obj_train_data/labels"  # Replace with the path to your labels folder
excess_folder = "hhdata_v0/obj_train_data/excess"  # Replace with the path to your excess folder

# Create the excess folder if it doesn't exist
os.makedirs(excess_folder, exist_ok=True)

# Iterate through all image files in the images folder
for image_file in os.listdir(images_folder):
    if image_file.endswith(('.jpg', '.png')):  # Process only image files
        # Generate the corresponding label file name
        label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(labels_folder, label_file)

        # Check if the corresponding label file exists
        if not os.path.exists(label_path):
            # Move the image file to the excess folder
            shutil.move(os.path.join(images_folder, image_file), os.path.join(excess_folder, image_file))
            print(f"Moved {image_file} to excess folder.")

print("Processing complete!")