import os

# Path to the directory containing the files
directory = "hhdata_v0/obj_train_data/excess/labels"  # Replace with the path to your directory

# Iterate through all files in the directory
for file in os.listdir(directory):
    # Check if "labels" appears in the filename
    if "labels" in file:
        # Replace "labels" with "images" in the filename
        new_name = file.replace("labels", "images")
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {file} -> {new_name}")

print("File renaming complete!")