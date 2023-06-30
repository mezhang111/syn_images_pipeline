import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", help="the root directory to search for files")
args = parser.parse_args()
for subdir, dirs, files in os.walk(args.root_dir):
    for file in files:
        # Split the filename into base name and extension
        base_name, ext = os.path.splitext(file)
        # Construct the new filename by adding 5000 to the base name and re-attaching the extension
        old_name = os.path.join(subdir, file)
        new_name = os.path.join(subdir, str(int(base_name) + 5000) + ext)
        # Rename the file
        os.rename(old_name, new_name)