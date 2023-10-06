import os

def renameFiles():
    input_directory = "./Green_NoBg/"

    for filename in os.listdir(input_directory):
        if filename.startswith("green_") and filename.endswith("_label.png"):
            base_filename, extension = os.path.splitext(filename)
            new_filename = "green_" + base_filename[5:] + extension
            old_path = os.path.join(input_directory, filename)
            new_path = os.path.join(input_directory, new_filename)
            os.rename(old_path, new_path)
            print("Renamed:", filename, "->", new_filename)

    print("All files renamed")

renameFiles()