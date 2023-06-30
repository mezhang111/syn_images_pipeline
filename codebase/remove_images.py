from PIL import Image
import os

folder_path = "/home/mengtao/Downloads/example_datasets/object_detection_example/yolo-chairs.v2i.yolov7pytorch/train/images"

# loop through all files in the folder
for filename in os.listdir(folder_path):
    filepath = os.path.join(folder_path, filename)
    
    # check if the file is an image
    if os.path.isfile(filepath) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        
        # open the image with Pillow
        with Image.open(filepath) as img:
            # check if the image has a height of 416 pixels
            if img.height == 416:
                # delete the image file
                os.remove(filepath)
                print(f"Deleted {filename}")
