import os
import shutil  # Define the paths to the source and destination folders

src_dir = '/home/mengtao/Downloads/example_datasets/object_detection_example/chair_close_ups_yolo'
dst_dir = '/home/mengtao/Downloads/example_datasets/object_detection_example/chair_furniture_light_with_occlusion_close_ups_yolo'  # Walk through the source directory and its subdirectories


for subdir, dirs, files in os.walk(src_dir):
    # Get the relative path of the current subdirectory with respect to the source directory
    rel_subdir = os.path.relpath(subdir, src_dir)
    # Construct the corresponding subdirectory in the destination directory
    dst_subdir = os.path.join(dst_dir, rel_subdir)
    # Create the subdirectory in the destination directory if it doesn't exist
    os.makedirs(dst_subdir, exist_ok=True)
    # Copy all files in the current subdirectory to the corresponding subdirectory in the destination directory
    for file in files:
        src_file = os.path.join(subdir, file)
        base_name, ext = os.path.splitext(file)
        try:
            dst_file = os.path.join(dst_subdir, str(int(base_name) + 10000) + ext)
        except ValueError:
            dst_file = os.path.join(dst_subdir, base_name + ext)
        shutil.copy(src_file, dst_file)
