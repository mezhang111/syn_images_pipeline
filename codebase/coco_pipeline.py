import os
import random
import shutil
import glob


# first delete empty annotations, then subsample, then split test from train, then create new label folder for single class


def check_class_in_annotation_file(file_path, class_ids):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        label_id, _, _, _, _ = line.strip().split(' ')
        if int(label_id) in class_ids:
            return True

    return False


def filter_dataset_by_class(label_path, class_ids, rename_class_ids=False):
    # Get all label files in the labels directory
    label_files = glob.glob(os.path.join(label_path, "*.txt"))
    class_ids = [str(c) for c in class_ids]
    # Create a mapping from kept class IDs to sequential indices starting from 0
    if not rename_class_ids:
        class_id_mapping = {str(class_id): str(class_id) for index, class_id in enumerate(class_ids)}
    else:
        class_id_mapping = {str(class_id): str(index) for index, class_id in enumerate(class_ids)}

    for label_file in label_files:
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


def delete_empty_annotations(dataset_root_path, probability, class_ids):
    images_path = os.path.join(dataset_root_path, 'images')
    labels_path = os.path.join(dataset_root_path, 'labels')

    image_files = os.listdir(images_path)
    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]
        label_file = image_name + '.txt'
        label_path = os.path.join(labels_path, label_file)

        # Check if the label file exists and if it is empty
        if os.path.exists(label_path):
            if not check_class_in_annotation_file(label_path, class_ids=class_ids):
                if random.random() < probability:
                    # Delete the image and label file
                    image_path = os.path.join(images_path, image_file)
                    os.remove(image_path)
                    os.remove(label_path)
                    print(f"Deleted empty annotations for {image_file}")


def subsample(source_dir, subsample_probability):
    # Create the destination directory if it doesn't exist
    # Iterate through the source directory
    image_folder = os.path.join(source_dir, 'images')
    label_dir = os.path.join(source_dir, 'labels')
    for filename in os.listdir(image_folder):
        # Generate a random number between 0 and 1
        random_num = random.uniform(0, 1)

        # Determine whether to keep or delete the image based on the probability
        if random_num >= subsample_probability:
            # Copy the image to the destination directory
            os.remove(os.path.join(image_folder, filename))
            image_name = os.path.splitext(filename)[0]
            label_file = image_name + '.txt'
            os.remove(os.path.join(label_dir, label_file))
    print("Subsampling complete.")


def split_test_from_train(original_dataset_folder, test_dataset_folder, test_prob):
    # Get a list of all image files in the 'images' subfolder of the original dataset
    original_images_folder = os.path.join(original_dataset_folder, 'images')
    image_files = [file for file in os.listdir(original_images_folder) if file.endswith('.jpg')]
    # Randomly select a portion of the dataset for the test set
    test_files = random.sample(image_files, int(len(image_files) * test_prob))
    test_image_folder = os.path.join(test_dataset_folder, 'images')
    test_label_folder = os.path.join(test_dataset_folder, 'labels')
    # Create test dataset folder
    os.makedirs(test_image_folder, exist_ok=True)
    os.makedirs(test_label_folder, exist_ok=True)
    # Move selected files to the test dataset folder
    for file in test_files:
        src_image = os.path.join(original_images_folder, file)
        src_label = os.path.join(original_dataset_folder, 'labels', file.replace('.jpg', '.txt'))
        dst_image = os.path.join(test_dataset_folder, 'images', file)
        dst_label = os.path.join(test_dataset_folder, 'labels', file.replace('.jpg', '.txt'))

        shutil.move(src_image, dst_image)
        shutil.move(src_label, dst_label)
    print("Test dataset created successfully!")


if __name__ == '__main__':
    class_ids = [13]
    remove_probability = 0.9
    subsample_probability = 0.5
    test_ratio = 0.5
    train_root_dir = '/home/mengtao/Downloads/microsoft_coco_bottle/train'
    test_root_dir = '/home/mengtao/Downloads/microsoft_coco_bottle/test'
    val_root_dir = '/home/mengtao/Downloads/microsoft_coco_bottle/valid'
    delete_empty_annotations(train_root_dir, remove_probability, class_ids)
    delete_empty_annotations(val_root_dir, remove_probability, class_ids)
    subsample(train_root_dir, subsample_probability)
    split_test_from_train(train_root_dir, test_root_dir, test_ratio)
    # make filtered labels
    roots = [train_root_dir, test_root_dir, val_root_dir]
    for r in roots:
        label_dir = os.path.join(r, 'labels')
        filtered_label_dir = os.path.join(r, 'filtered_labels')
        shutil.copytree(label_dir, filtered_label_dir, dirs_exist_ok=True)
        filter_dataset_by_class(filtered_label_dir, class_ids, rename_class_ids=False)
        if r != train_root_dir: # rename folders for test and val
            os.rename(label_dir, os.path.join(r, 'labels_multiclass'))
            os.rename(filtered_label_dir, os.path.join(r, 'labels'))