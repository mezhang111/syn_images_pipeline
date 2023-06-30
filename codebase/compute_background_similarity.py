from metadata import get_background_similarity
import os
import json
import pandas as pd
from pycocotools import mask as maskUtils
root_dir = "/home/mengtao/Downloads/example_datasets/object_detection_example" \
           "/chair_furniture_light_with_background_styled"


def find_file_with_substring(folder_path, substring):
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if substring in filename:
                splits = dirpath.split("/")
                if splits[-1] == "images":
                    return os.path.join(dirpath, filename), splits[-2]
    return None


def get_background_similarities_for_image(annotations, image_path):
    background_similarities = []
    for idx, ann in enumerate(annotations):
        segmentation = ann["segmentation"]
        height = ann["height"]
        width = ann["width"]
        mask = maskUtils.frPyObjects(segmentation, height, width)
        mask = maskUtils.decode(mask)  # binary mask of 0, 1
        similarity, _, _ = get_background_similarity(mask, image_path)
        background_similarities.append(similarity)
    return background_similarities


def main():
    coco_annotation_path = os.path.join(root_dir, "coco_annotations.json")
    with open(coco_annotation_path, 'r', encoding="utf-8") as fp:
        coco_annotations = json.load(fp)
    all_annotations = coco_annotations["annotations"]
    image_ids = [image["id"] for image in coco_annotations["images"]]
    annotations_split_in_image = []
    metadata_train = {"background_similarity": [], "image_id": []}
    metadata_valid = {"background_similarity": [], "image_id": []}
    metadata_test = {"background_similarity": [], "image_id": []}
    metadatas = {"train": metadata_train, "valid": metadata_valid, "test": metadata_test}

    for _id in image_ids:
        image_anns = [ann for ann in all_annotations if ann["image_id"] == _id]
        annotations_split_in_image.append(image_anns)
    for annotation_per_img, img_id in zip(annotations_split_in_image, image_ids):
        image_id = str(img_id).zfill(6)
        image_path, split = find_file_with_substring(root_dir, image_id)
        background_similarities = get_background_similarities_for_image(annotation_per_img, image_path)
        metadatas[split]["image_id"].append(image_id)
        metadatas[split]["background_similarity"].append(background_similarities)

    for split, metadata in metadatas.items():
        df = pd.DataFrame(metadata)
        metadata_path = os.path.join(root_dir, split, "metadata.csv")
        df.to_csv(metadata_path, mode="a+", index=False)


if __name__ == "__main__":
    main()