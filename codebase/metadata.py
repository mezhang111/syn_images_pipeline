import numpy as np
import os
import json
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from pycocotools import mask as maskUtils
from PIL import Image
from scipy import ndimage

metadata_columns = ["image_id", "obj_area", "category_id", "name", "instance_id", "depth",
                    "depth_area_product", "background_similarity", "bbox_std"]

root_dir = "/home/mengtao/experiments/generated_dataset/pistol_weapon_on_surface/coco_data"
bbox_area_threshold = 1536
filter_out_small_bboxes = True
# workflow: first metadata.py, then random_backgrounds/coco_to_yolo_split.py


def update_metadata(metadata, annotations, instances, depth_image):
    names = list(instances["name"])
    instance_ids = list(instances["id"])
    image_ids = []
    category_ids = []
    depths = []
    obj_areas = []
    depth_area_products = []
    background_similarities = []
    bbox_stds = []
    local_background_vectors = [] # list of local background vectors around each object
    local_obj_vectors = [] # list of object boundary vectors
    to_remove_idx = []
    for idx, ann in enumerate(annotations):
        if filter_out_small_bboxes and ann["area"] < bbox_area_threshold:
            to_remove_idx.append(idx)
            continue
        segmentation = ann["segmentation"]
        height = ann["height"]
        width = ann["width"]
        mask = maskUtils.frPyObjects(segmentation, height, width)
        mask = maskUtils.decode(mask)  # binary mask of 0, 1
        image_id = str(ann["image_id"]).zfill(6)
        image_ids.append(image_id)
        category_ids.append(ann["category_id"])
        obj_area = np.count_nonzero(mask == 1)
        obj_indices = np.argwhere(mask == 1)
        avg_depth = 0.
        for x, y in obj_indices:
            avg_depth += depth_image[x, y]
        avg_depth /= obj_area
        depths.append(avg_depth)
        obj_areas.append(obj_area)
        #  The idea is area = c/depth, if no occlusion. So if depth*area can somehow indicate the occlusion
        denominator = height * width / 10.
        depth_area_products.append(avg_depth * obj_area / denominator)
        image_filename = image_id + ".png"
        image_path = os.path.join(root_dir, "images", image_filename)
        similarity, background_vector, obj_vector = get_background_similarity(mask, image_path)
        local_background_vectors.append(background_vector)
        local_obj_vectors.append(obj_vector)
        background_similarities.append(similarity)
        x1, y1, x2, y2 = get_bbox_ranges_from_coco(*ann["bbox"])
        bbox_stds.append(get_bbox_std(image_path, x1, y1, x2, y2))
    if filter_out_small_bboxes:
        for idx in reversed(to_remove_idx):
            names.pop(idx)
            instance_ids.pop(idx)
    metadata["image_id"].append(image_ids)
    metadata["category_id"].append(category_ids)
    metadata['name'].append(names)
    metadata['instance_id'].append(instance_ids)
    metadata["depth"].append(depths)
    metadata["obj_area"].append(obj_areas)
    metadata["depth_area_product"].append(depth_area_products)
    metadata["background_similarity"].append(background_similarities)
    metadata["bbox_std"].append(bbox_stds)

    return local_background_vectors, local_obj_vectors


def read_image(image_path, normalize=True):
    image = Image.open(image_path)
    data = np.array(image)
    if normalize:
        data = (data-127.5)/127.5 # normalize to [-1,1]
    return data


def get_background_similarity(binary_mask, image_path, margin=2):
    data = read_image(image_path, normalize=True)
    if np.all(binary_mask == 1):
        binary_mask[0, 0] = 0
    object_mask = binary_mask==1
    background_mask = binary_mask==0
    kernel_structure = ndimage.generate_binary_structure(2, 3) # mask for dilation
    # Create a binary mask that is True for pixels that are adjacent to the background
    obj_adjacent_mask = ndimage.binary_dilation(background_mask, structure=kernel_structure, iterations=1) & object_mask
    adj_indices = np.argwhere(obj_adjacent_mask == 1)
    dissimilarity = 0.0
    background_means = []
    obj_values = []
    for (i, j) in adj_indices:
        local_background_indices = get_local_neighbouring_backgrounds(background_mask, margin, i, j)
        local_background_mean, local_obj_value = compute_local_dissimilarities(data, (i,j), local_background_indices)
        background_means.append(local_background_mean)
        obj_values.append(local_obj_value)
    background_means = np.concatenate(background_means)
    obj_values = np.concatenate(obj_values)
    similarity = np.unique(cosine_similarity([background_means], [obj_values]))[0]
    # similarity = np.exp(0.5*(-dissimilarity**2))
    return similarity, background_means, obj_values


def compute_local_dissimilarities(image_data, obj_indices, background_indices):
    background_mean = 0.0
    euclidian_distance = 0.0
    obj_i = obj_indices[0]
    obj_j = obj_indices[1]
    weight_sum = 0.0
    for (i,j) in background_indices:
        # euclidian_distance += np.sum((image_data[obj_indices[0], obj_indices[1], :] - image_data[i, j, :])**2)
        # weight = 1/np.sqrt((i-obj_i)**2 + (j-obj_j)**2)
        weight = 1.0
        weight_sum += weight
        background_mean += weight*image_data[i, j, :]
    background_mean = background_mean/weight_sum
    obj_value = image_data[obj_i, obj_j]
    #euclidian_distance /= len(background_indices)
    return background_mean, obj_value 
    

def get_local_neighbouring_backgrounds(background_mask, margin, i, j):
    offsets = [i-margin for i in range(2*margin+1)]
    width, height = background_mask.shape
    indices = []
    for p in offsets:
        curr_i = i+p
        for q in offsets:
            curr_j = j+q
            if (0 <= curr_i < width) and (0 <= curr_j < height) and background_mask[curr_i, curr_j]:
                indices.append((curr_i, curr_j))
    return indices


def get_bbox_ranges_from_coco(x1, y1, width, height):
    return x1, y1, x1+width, y1+height


def get_bbox_std(image_path, x1, y1, x2, y2):
    image_data = read_image(image_path, normalize=True)
    roi = image_data[x1:x2, y1:y2, :]
    std = np.std(roi, axis=(0,1)) # calculate standard deviation for each color
    return np.sum(std)


def main():
    coco_annotation_path = os.path.join(root_dir, "coco_annotations.json")
    instance_info_path = os.path.join(root_dir, "instance.csv")
    metadata_path = os.path.join(root_dir, "metadata.csv")
    depth_path = os.path.join(root_dir, "depth.npy")
    metadata = {key: [] for key in metadata_columns}
    with open(coco_annotation_path, 'r', encoding="utf-8") as fp:
        coco_annotations = json.load(fp)
    instances = pd.read_csv(instance_info_path, converters={key: pd.eval for key in pd.read_csv(instance_info_path, nrows=1).columns})
    depths = np.load(depth_path)
    image_ids = [image["id"] for image in coco_annotations["images"]]
    all_annotations = coco_annotations["annotations"]
    annotations_split_in_image = []
    counter = 0
    background_per_image = [] # one background/obj vector per image
    obj_per_image = []
    for _id in image_ids:
        image_anns = [ann for ann in all_annotations if ann["image_id"] == _id]
        annotations_split_in_image.append(image_anns)
        counter += len(image_anns)

    for idx, annotation_per_image in enumerate(annotations_split_in_image):
        depth = depths[idx]
        instance_per_image = instances.iloc[idx]
        background, obj = update_metadata(metadata, annotation_per_image, instance_per_image, depth)
        background_per_image.append(background)
        obj_per_image.append(obj)
    #print(metadata["image_id"])
    df = pd.DataFrame(metadata)
    df.to_csv(metadata_path, mode="a+", index=False)
    with open(os.path.join(root_dir, "backgrounds.pickle"), 'wb') as f:
        pickle.dump(background_per_image, f)
    with open(os.path.join(root_dir, "objects.pickle"), 'wb') as f:
        pickle.dump(obj_per_image, f)


if __name__ == "__main__":
    main()
