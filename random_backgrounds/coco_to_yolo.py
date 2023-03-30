import os
import json
import random
import numpy as np
from pycocotools.coco import COCO

# path to the COCO annotation file
coco_annotation_path = "path/to/coco/annotations.json"
# path to the image folder
image_folder_path = "path/to/images"
# output folder for YOLO annotations
yolo_annotation_folder = "path/to/yolo/annotations"
# output folder for train, valid, and test splits
train_folder = "path/to/train/folder"
valid_folder = "path/to/valid/folder"
test_folder = "path/to/test/folder"
# probability for train, valid, and test splits
train_prob = 0.7
valid_prob = 0.2
test_prob = 0.1

# create output folders if they don't exist
if not os.path.exists(yolo_annotation_folder):
    os.makedirs(yolo_annotation_folder)
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(valid_folder):
    os.makedirs(valid_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# load COCO annotations
coco = COCO(coco_annotation_path)

# load category names
category_names = {}
for category in coco.loadCats(coco.getCatIds()):
    category_names[category['id']] = category['name']

# get image IDs
image_ids = coco.getImgIds()

# shuffle image IDs
random.shuffle(image_ids)

# split image IDs
num_images = len(image_ids)
train_end_index = int(num_images * train_prob)
valid_end_index = train_end_index + int(num_images * valid_prob)
train_image_ids = image_ids[:train_end_index]
valid_image_ids = image_ids[train_end_index:valid_end_index]
test_image_ids = image_ids[valid_end_index:]


# convert COCO annotations to YOLO annotations and save to file
def convert_annotations(image_id, output_file):
    # load image
    image_info = coco.loadImgs(image_id)[0]
    image_filename = os.path.join(image_folder_path, image_info['file_name'])
    image_width = image_info['width']
    image_height = image_info['height']

    # load annotations
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=image_id))

    # convert annotations to YOLO format
    for annotation in annotations:
        category_id = annotation['category_id']
        category_name = category_names[category_id]
        x, y, w, h = annotation['bbox']
        x_center = x + w / 2
        y_center = y + h / 2
        x_center /= image_width
        y_center /= image_height
        w /= image_width
        h /= image_height
        line = f"{category_name} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"
        output_file.write(line)


# loop through train, valid, and test image IDs and convert annotations
for split_name, image_ids in [("train", train_image_ids), ("valid", valid_image_ids), ("test", test_image_ids)]:
    with open(os.path.join(yolo_annotation_folder, f"{split_name}.txt"), "w") as output_file:
        for image_id in image_ids:
            convert_annotations(image_id, output_file)

            # copy image to appropriate split folder
            image_filename = coco.loadImgs(image_id)[0]['file_name']
            input_path = os.path.join(image_folder_path, image_filename)
            if split_name == "train":
                output_path = os.path.join(train_folder, image_filename)
            elif split_name == "valid":
                output_path = os.path.join(valid_folder, image_filename)
            else:
                output_path = os.path.join(test_folder, image_filename)
            os.system(f"cp {input_path} {output_path}")

