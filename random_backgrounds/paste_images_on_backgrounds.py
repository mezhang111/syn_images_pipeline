#!/usr/bin/env python3
"""This script allows to automatically paste generated images on random backgrounds."""

import os
import random
import argparse
from PIL import Image


def main():
    # Get and parse all given arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--images",
                        default='random_backgrounds/chairs_no_random_rotation/coco_data/images',
                        type=str,
                        help="Path to object images to paste.")
    parser.add_argument("-b",
                        "--backgrounds",
                        default='random_backgrounds/backgrounds',
                        type=str,
                        help="Path to background images to paste on.")
    parser.add_argument("-t",
                        "--types",
                        default=('jpg', 'jpeg', 'png'),
                        type=str,
                        nargs='+',
                        help="File types to consider. Default: jp[e]g, png.")
    parser.add_argument(
        "--output",
        "-o",
        default="random_backgrounds/chairs_no_rotation_with_background/coco_data",
        type=str,
        help=
        "Merges images and backgrounds, overwriting original files. Default: False."
    )

    args = parser.parse_args()

    # Create an output directory if `overwrite` is not selected
    image_folder = os.path.join(args.output, "images")
    os.makedirs(image_folder, exist_ok=True)

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

            # save
            background.save(os.path.join(image_folder, file_name))




if __name__ == "__main__":
    main()
