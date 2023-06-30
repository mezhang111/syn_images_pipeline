import os
import glob
import random


def filter_dataset_by_class(dataset_dir, class_ids, p):
    # Get all label files in the labels directory
    label_files = glob.glob(os.path.join(dataset_dir, "labels", "*.txt"))

    # Create a mapping from kept class IDs to sequential indices starting from 0
    class_id_mapping = {class_id: class_id for index, class_id in enumerate(class_ids)}

    for label_file in label_files:
        # Get the corresponding image file name
        image_file = os.path.join(dataset_dir, "images", os.path.splitext(os.path.basename(label_file))[0] + ".jpg")

        # Read the contents of the label file
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # Filter out the bounding boxes for the specified class IDs and update class IDs with mapped indices
        filtered_lines = []
        for line in lines:
            class_id, *rest = line.split()
            if class_id in class_ids:
                mapped_class_id = class_id_mapping[class_id]
                filtered_lines.append(mapped_class_id + " " + " ".join(rest) + "\n")

        with open(label_file, 'w') as f:
            f.writelines(filtered_lines)
        if len(filtered_lines) == 0:
            # If no bounding boxes are left, randomly remove the label file and the corresponding image file with probability p
            if random.random() < p:
                os.remove(label_file)
                os.remove(image_file)


dataset_directory = "/home/mengtao/Downloads/microsoft_coco_car/train"
class_ids_to_keep = ['18']
removal_probability = 0.8  # Example: 97.5% chance of removal

filter_dataset_by_class(dataset_directory, class_ids_to_keep, removal_probability)
