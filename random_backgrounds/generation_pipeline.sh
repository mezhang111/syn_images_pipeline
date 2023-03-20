#!/bin/bash
# bash generation_pipeline.sh  object_foler_path  output_dir_for_synthetic_images num num_camera background_folder output_background project create output
# read arguments
object_folder="$1"
syntethic_image_folder="$2"
num_poses="$3"
num_cameras="$4"
background_folder="$5"
output_background="$6"
project="$7"
create="$8"
download_folder="$9"

blenderproc run random_backgrounds/generate_images.py --scene "${object_folder}" --output_dir "${syntethic_image_folder}" --num "${num_poses}" --cameras "${num_cameras}"

python random_backgrounds/paste_images_on_backgrounds.py --images "${syntethic_image_folder}"/coco_data/images --backgrounds "${background_folder}" --output "${output_background}"

python random_backgrounds/to_bbox_annotations.py --file "${syntethic_image_folder}"/coco_data/coco_annotations.json --output "${output_background}"

if [ "$create" = "true" ] || [ "$create" = "True" ]; then
  python random_backgrounds/upload_download_roboflow.py --project "${project}" --folder "${output_background}" --output "${download_folder}" --create
else
  python random_backgrounds/upload_download_roboflow.py --project "${project}" --folder "${output_background}" -output "${download_folder}"
fi

