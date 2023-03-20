import json
import argparse
import os
# Remove the segmentation mask from the coco annotations


parser = argparse.ArgumentParser()
parser.add_argument('--output', default="random_backgrounds/images_with_backgrounds_test/coco_data",
                    help="Path to where the final files will be saved")
parser.add_argument("--file", default='random_backgrounds/output_test/coco_data/coco_annotations.json',
                    type=str, help="Path to the coco file")
args = parser.parse_args()
with open(args.file) as json_file:
    coco = json.load(json_file)
annotations = coco['annotations']
for annotation in annotations:
    annotation.pop('segmentation', None)

with open(os.path.join(args.output, 'coco_annotations.json'), 'a+') as outfile:
    json.dump(coco, outfile)
