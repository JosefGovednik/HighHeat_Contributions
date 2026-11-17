import os

# Path to the parent directory containing the datasets
datasets_folder = "hhdata_v1\obj_train_data\excess"  # Replace with the path to your datasets folder

# Iterate through each dataset directory
for dataset in os.listdir(datasets_folder):
    dataset_path = os.path.join(datasets_folder, dataset)

    # Skip if it's not a directory
    if not os.path.isdir(dataset_path):
        continue

    # Iterate through all image files in the dataset
    for file in os.listdir(dataset_path):
        if file.endswith('.png') or file.endswith('.txt'):  # Process only .png and .txt files
            # Construct the new filename with the dataset name as a suffix
            name, ext = os.path.splitext(file)
            new_name = f"{name}_{dataset}{ext}"
            old_path = os.path.join(dataset_path, file)
            new_path = os.path.join(dataset_path, new_name)

            # Rename the file
            os.rename(old_path, new_path)

print("File renaming complete!")