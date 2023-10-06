import os
import shutil

def moveAndRenameImages():
    source_directories = ["apple", "capsicium", "greenapple", "mango", "orange"]
    destination_directory = "./images/"

    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    image_count = 1

    for source_dir in source_directories:
        source_path = os.path.join(os.getcwd(), source_dir)  # Get the full path of the source directory
        for filename in os.listdir(source_path):
            if filename.endswith(".png"):
                source_file_path = os.path.join(source_path, filename)
                destination_file_path = os.path.join(destination_directory, f"image_{image_count}.png")
                shutil.move(source_file_path, destination_file_path)
                print("Moved:", source_file_path, "->", destination_file_path)
                image_count += 1

    print("All images moved and renamed")

moveAndRenameImages()
