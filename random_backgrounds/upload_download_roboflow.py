import os
import requests
import base64
import io
import argparse
import random
from roboflow import Roboflow
from PIL import Image

MY_KEY = "W389caWDMxj3pEuYdHuv"
parser = argparse.ArgumentParser()
parser.add_argument('--project', type=str, required=True, help='name of the project in roboflow workspace')
parser.add_argument('--folder', type=str, default='random_backgrounds/images_with_backgrounds_test/coco_data',
                    help='Path to coco folder')
parser.add_argument('--create', action='store_true',
                    help='create new project')
parser.add_argument('--output', type=str, required=False, help='path to the output dir od the downloaded folder')

args = parser.parse_args()
image_folder = os.path.join(args.folder, 'images')
annotation_path = os.path.join(args.folder, 'coco_annotations.json')
rf = Roboflow(api_key=MY_KEY)
workspace = rf.workspace()
if args.create:
    project = workspace.create_project(project_name=args.project, project_type="object-detection", annotation="objects",
                                       project_license="MIT")
else:
    project = workspace.project(args.project)
# project.upload(image_folder, os.path.join(args.folder, 'coco_annotations.json'), num_retry_uploads=10)


for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        print(filename)
        image_path = os.path.join(image_folder, filename)
        split = random.choices(['train', 'valid', 'test'], [0.7, 0.15, 0.15], k=1)[0]
        project.upload(image_path, annotation_path, split=split, num_retry_uploads=10)

# generate base version
config = {
            "augmentation": {

            },
            "preprocessing": {
                "auto-orient": True,
            }
        }
version_number = project.generate_version(config)
if args.output:
    os.makedirs(args.output, exist_ok=True)
    project.version(version_number=version_number).download(model_format="yolov5", location=args.output)
