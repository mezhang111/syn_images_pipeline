import os
import json
import random
import shutil
from pathlib import Path
import argparse

bbox_area_threshold = 1536


def create_yolo_annotation(coco_annotation, img_width, img_height, filter_out_small_bboxes):
    if filter_out_small_bboxes and coco_annotation['area'] < bbox_area_threshold: # filter it out
        return None
    category_id = coco_annotation['category_id'] - 1  # 0 is reserved in coco
    x_center = (coco_annotation['bbox'][0] + coco_annotation['bbox'][2] / 2) / img_width
    y_center = (coco_annotation['bbox'][1] + coco_annotation['bbox'][3] / 2) / img_height
    width = coco_annotation['bbox'][2] / img_width
    height = coco_annotation['bbox'][3] / img_height

    yolo_annotation = f"{category_id} {x_center} {y_center} {width} {height}"
    return yolo_annotation


def split_coco_to_yolo(image_folder, coco_annotation_path, output_folder, probs, filter_out_small_bboxes):
    with open(coco_annotation_path, "r") as f:
        coco_data = json.load(f)

    annotations = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations:
            annotations[image_id] = []
        annotations[image_id].append(annotation)

    # Create output directories
    output_folder = Path(output_folder)
    train_folder = output_folder / 'train'
    val_folder = output_folder / 'valid'
    test_folder = output_folder / 'test'

    for folder in [train_folder, val_folder, test_folder]:
        os.makedirs(str(folder / "images"), exist_ok=True)
        os.makedirs(str(folder / "labels"), exist_ok=True)
        print(str(folder / "images"))

    image_ids = list(annotations.keys())
    random.shuffle(image_ids)

    num_images = len(image_ids)
    train_ratio, val_ratio, test_ratio = probs
    num_train = int(train_ratio * num_images)
    num_val = int(val_ratio * num_images)

    train_ids = image_ids[:num_train]
    val_ids = image_ids[num_train:num_train + num_val]
    test_ids = image_ids[num_train + num_val:]

    for image_data in coco_data['images']:
        image_id = image_data['id']
        image_filename = image_data['file_name']
        image_path = os.path.join(image_folder, image_filename)

        if image_id not in annotations:
            yolo_annotations = []
        else:
            img_width = image_data['width']
            img_height = image_data['height']
            yolo_annotations = [
                create_yolo_annotation(ann, img_width, img_height, filter_out_small_bboxes)
                for ann in annotations[image_id]
            ]
            yolo_annotations = [ann for ann in yolo_annotations if ann is not None]

        if image_id in train_ids:
            target_folder = train_folder
        elif image_id in val_ids:
            target_folder = val_folder
        else:
            target_folder = test_folder

        target_image_path = target_folder / image_filename if image_filename.split('/')[0] == 'images' else target_folder / 'images' / image_filename
        shutil.copyfile(image_path, target_image_path)

        target_annotation_path = target_folder / "labels" / (Path(image_filename).stem + '.txt')
        with open(target_annotation_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Path to the coco folder")
    parser.add_argument("--annotation", type=str, default=None, help="Path to the coco annotation")
    parser.add_argument("--output", type=str, required=True, help="Path to the output folder")
    parser.add_argument('--split', nargs='+', required=True, help='--split train valid test')
    parser.add_argument('--filter', action='store_true', help='if you wanna filter out small bboxes')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    image_folder = args.folder
    if args.annotation is None:
        coco_annotation_path = args.folder + "/coco_annotations.json"
    else:
        coco_annotation_path = args.annotation
    probs = [float(s) for s in args.split]
    split_coco_to_yolo(image_folder=image_folder, coco_annotation_path=coco_annotation_path, output_folder=args.output,
                       probs=probs, filter_out_small_bboxes=args.filter)
