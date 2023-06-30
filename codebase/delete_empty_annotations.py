import os
import random
import shutil


def check_class_in_annotation_file(file_path, class_id):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        label_id, _, _, _, _ = line.strip().split(' ')
        if int(label_id) == class_id:
            return True

    return False


def delete_empty_annotations(dataset_root_path, probability):
    images_path = os.path.join(dataset_root_path, 'images')
    labels_path = os.path.join(dataset_root_path, 'labels')

    image_files = os.listdir(images_path)
    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]
        label_file = image_name + '.txt'
        label_path = os.path.join(labels_path, label_file)

        # Check if the label file exists and if it is empty
        if os.path.exists(label_path):
            if not check_class_in_annotation_file(label_path, class_id=48):
                if random.random() < probability:
                    # Delete the image and label file
                    image_path = os.path.join(images_path, image_file)
                    os.remove(image_path)
                    os.remove(label_path)
                    print(f"Deleted empty annotations for {image_file}")


# Example usage
dataset_root = '/home/mengtao/Downloads/microsoft_coco_person/valid'
deletion_probability = 0.7
delete_empty_annotations(dataset_root, deletion_probability)
