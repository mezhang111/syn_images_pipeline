import os
import cv2
import numpy as np
import pandas as pd
from metadata import get_bbox_std, read_image
from sklearn.metrics.pairwise import cosine_similarity

root_dir = "/home/mengtao/Downloads/example_datasets/object_detection_example/Real Chairs.v5-300-70-129.yolov5pytorch"


def convert_yolo_to_pascal_voc(label_path, image_path):
    # read label and image files
    label_file = open(label_path, 'r')
    lines = label_file.readlines()
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    # initialize lists to store bounding box information
    bboxes = []
    classes = []

    # iterate over each line in the label file and extract bounding box coordinates
    for line in lines:
        data = line.split()
        class_id = int(data[0]) + 1  # align with coco convention
        x_center = float(data[1]) * width
        y_center = float(data[2]) * height
        bbox_width = float(data[3]) * width
        bbox_height = float(data[4]) * height

        # convert bounding box from YOLO format to Pascal VOC format
        xmin = int(round(x_center - (bbox_width / 2)))
        ymin = int(round(y_center - (bbox_height / 2)))
        xmax = int(round(x_center + (bbox_width / 2)))
        ymax = int(round(y_center + (bbox_height / 2)))

        # append bounding box coordinates and class id to lists
        bboxes.append([xmin, ymin, xmax, ymax])
        classes.append(class_id)

    # return bounding boxes and classes in Pascal VOC format
    return bboxes, classes


def _mean_std_of_box(image_path, bbox_coordinates, step_x, step_y):
    # extract the bounding box from the image
    x1, y1, x2, y2 = bbox_coordinates

    # split the bounding box into smaller boxes
    boxes = [(x, y, x + step_x, y + step_y)
             for x in range(x1, x2 - step_x + 1, step_x) for y in range(y1, y2 - step_y + 1, step_y)]

    # compute the std of each smaller box
    stds = [get_bbox_std(image_path, x1, y1, x2, y2) for (x1, y1, x2, y2) in boxes]

    # take the mean of std
    sorted_stds = np.sort(stds)[::-1]
    mean_std = np.mean(sorted_stds[:10])
    return mean_std


def compute_mean_std_of_box(image_path, bbox):
    step_x, step_y = _get_step_size(bbox, 4)
    return _mean_std_of_box(image_path, bbox, step_x, step_y)


def _get_step_size(bbox, k):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    step_x = max(int(width / k), 1)
    step_y = max(int(height / k), 1)
    return step_x, step_y


def compute_background_similarity(image_path, bbox):
    image = read_image(image_path)
    width, height = image.shape[:2]
    step_x, step_y = _get_step_size(bbox, 4)
    x1, y1, x2, y2 = bbox
    if x2-x1 < 10 or y2-y1 < 10:
        return 1
    boxes = [(x1 + step_x, y1, x2 - step_x, y2 - step_y), (x1 + step_x, y1 + step_y, x2, y2 - step_y),
             (x1 + step_x, y1 + step_y, x2 - step_x, y2), (x1, y1 + step_y, x2 - step_x, y2 - step_y)]
    background_boxes = [(x1, y1 - 3, x2, y1), (x2, y1, x2 + 3, y2),
                        (x1, y2, x2, y2 + 3), (x1 - 3, y1, x1, y2)]
    object_vector = []
    background_vector = []
    for x_1, y_1, x_2, y_2 in boxes:
        object_vector.append(np.mean(image[x_1:x_2, y_1:y_2], axis=(0, 1)))
    keep_indices = []
    for idx, (x_1, y_1, x_2, y_2) in enumerate(background_boxes):
        if x_1 < 0 or x_2 > width or y_1 < 0 or y_2 > height:
            background_vector.append(None)
        else:
            background_vector.append(np.mean(image[x_1:x_2, y_1:y_2], axis=(0, 1)))
            keep_indices.append(idx)
    if len(keep_indices) == 0:
        return 0
    object_vector = np.concatenate([v for idx, v in enumerate(object_vector) if idx in keep_indices])
    background_vector = np.concatenate([v for idx, v in enumerate(background_vector) if idx in keep_indices])
    similarity = np.unique(cosine_similarity([background_vector], [object_vector]))[0]
    return similarity


def write_bbox_std(base_path):
    # iterate over train, valid, and test subfolders
    for subdir in ['train', 'valid', 'test']:
        image_folder_path = os.path.join(base_path, subdir, 'images')
        label_folder_path = os.path.join(base_path, subdir, 'labels')
        metadata_path = os.path.join(base_path, subdir, 'metadata.csv')
        bbox_stds = []
        bbox_areas = []
        average_bbox_stds = []
        image_filenames = []
        classes_list = []
        similarity = []
        # iterate over image files in each subfolder
        for image_file_name in os.listdir(image_folder_path):
            image_path = os.path.join(image_folder_path, image_file_name)
            label_path = os.path.join(label_folder_path, os.path.splitext(image_file_name)[0] + '.txt')

            # convert YOLO format bounding boxes to Pascal VOC format
            bboxes, classes = convert_yolo_to_pascal_voc(label_path, image_path)
            image_filenames.append(image_file_name)
            bbox_stds.append([get_bbox_std(image_path, bbox[0], bbox[1], bbox[2], bbox[3]) for bbox in bboxes])
            bbox_areas.append([(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes])
            classes_list.append(classes)
            average_bbox_stds.append([compute_mean_std_of_box(image_path, bbox) for bbox in bboxes])
            similarity.append([compute_background_similarity(image_path, bbox) for bbox in bboxes])
        # add bounding box information to dictionary
        df = pd.DataFrame({"bbox_std": bbox_stds, "image_filename": image_filenames,
                           "category_id": classes_list, "bbox_area_custom": bbox_areas,
                           "average_bbox_std": average_bbox_stds, "quasi_similarity": similarity})
        df = df.sort_values(by=['image_filename'])  # order by image filename
        df.to_csv(metadata_path, mode="a+", index=False)


if __name__ == "__main__":
    write_bbox_std(root_dir)
