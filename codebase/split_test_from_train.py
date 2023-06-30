import os
import random
import shutil


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
    print("Creating test dataset...")
    # Path to the original YOLO dataset folder
    original_dataset_folder = '/home/mengtao/Downloads/microsoft_coco_person/train'

    # Path to the new test dataset folder
    test_dataset_folder = '/home/mengtao/Downloads/microsoft_coco_person/test'

    # Probability of selecting an image for the test set
    test_prob = 0.5
    split_test_from_train(original_dataset_folder, test_dataset_folder, test_prob)


