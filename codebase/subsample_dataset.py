import os
import random
import shutil


def subsample_images(source_dir, dest_dir, probability):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Iterate through the source directory
    for filename in os.listdir(source_dir):
        # Generate a random number between 0 and 1
        random_num = random.uniform(0, 1)

        # Determine whether to keep or delete the image based on the probability
        if random_num <= probability:
            # Copy the image to the destination directory
            shutil.copy2(os.path.join(source_dir, filename), dest_dir)

    print("Subsampling complete.")


source_directory = "/home/mengtao/Downloads/microsoft_coco_person/train/images"
destination_directory = "/home/mengtao/Downloads/microsoft_coco_person/train/tmp"
probability_to_keep = 0.1

subsample_images(source_directory, destination_directory, probability_to_keep)
