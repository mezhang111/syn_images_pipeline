"""This script allows to automatically paste generated images on random backgrounds."""
import shutil
import os
import random
import argparse
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path

def main():
    # Get and parse all given arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--images",
                        default='../generated_dataset/output_test_p1/coco_data/images',
                        type=str,
                        help="Path to object images to paste.")
    parser.add_argument("-b",
                        "--backgrounds",
                        default='../assets/pistol_background',
                        type=str,
                        help="Path to background images to paste on.")
    parser.add_argument("-t",
                        "--types",
                        default=('jpg', 'jpeg', 'png'),
                        type=str,
                        nargs='+',
                        help="File types to consider. Default: jp[e]g, png.")
    parser.add_argument("--metadata",
                        action='store_true',
                        help="Whether to copy metadata.csv and coco_annotations.json or not"
                        )
    parser.add_argument(
        "--output",
        "-o",
        default="../generated_dataset/pistol_p1/coco_data",
        type=str,
        help="output folder"
    )
    parser.add_argument("--blur", default=None, type=int, help="Blur the image in the end with given value")
    parser.add_argument("--p", default=0.5, type=float, help="Probability to blur the image")

    args = parser.parse_args()

    # Create an output directory if `overwrite` is not selected
    image_folder = os.path.join(args.output, "images")
    os.makedirs(image_folder, exist_ok=True)
    parent_folder_path = str(Path(args.images).parent.absolute())

    if args.metadata: # if metadata exists, copy
        for file_name in os.listdir(parent_folder_path):
            # Check if the file is a regular file (not a directory)
            file_path = os.path.join(parent_folder_path, file_name)
            if os.path.isfile(file_path):
                shutil.move(file_path, os.path.join(args.output, file_name))

    # Go through all files in given `images` directory
    for file_name in os.listdir(args.images):
        # Matching files to given `types` and opening images
        if file_name.lower().endswith(args.types):
            img_path = os.path.join(args.images, file_name)
            img = Image.open(img_path)
            img_w, img_h = img.size

            '''
            # using all backgrounds
            backgrounds = []
            for p in os.listdir(args.backgrounds):
                if p.lower().endswith(args.types):
                    background_path = os.path.join(args.backgrounds, p)
                    background = Image.open(background_path).resize([img_w, img_h])
                    backgrounds.append(background)

            for background in backgrounds:
                # Pasting the current image on the selected background
                background.paste(img, mask=img.convert('RGBA'))
                output_filename = f'img{counter}.png'
                background.save(os.path.join(args.output, output_filename))
                counter += 1

            '''
            # Selecting and opening a random image file from given `backgrounds` directory to use as background
            background_path = random.choice([
                os.path.join(args.backgrounds, p)
                for p in os.listdir(args.backgrounds)
                if p.lower().endswith(args.types)
            ])
            background = Image.open(background_path).resize([img_w, img_h])
            # Pasting the current image on the selected background
            background.paste(img, mask=img.convert('RGBA'))
            if args.blur is not None and args.blur > 1:
                if np.random.uniform(0, 1) <= args.p:
                    background = background.filter(ImageFilter.GaussianBlur(args.blur))

            # save
            background.save(os.path.join(image_folder, file_name))




if __name__ == "__main__":
    main()
