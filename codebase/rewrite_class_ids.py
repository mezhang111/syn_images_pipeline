import os

def rewrite_class_id(annotation_path, new_class_id):
    # Read the contents of the annotation file
    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    # Modify lines with class ID 0
    modified_lines = []
    for line in lines:
        data = line.strip().split()
        class_id = int(data[0])
        if class_id == 0:
            data[0] = str(new_class_id)
        modified_lines.append(' '.join(data) + '\n')

    # Write the modified lines back to the annotation file
    with open(annotation_path, 'w') as f:
        f.writelines(modified_lines)

# Path to the directory containing annotation files
annotation_directory = '/home/mengtao/Downloads/microsoft_coco_car_baseline2/train/labels'

# New class ID to replace class ID 0
new_class_id = 18

# Iterate through annotation files in the directory
for filename in os.listdir(annotation_directory):
    if filename.endswith('.txt'):
        annotation_path = os.path.join(annotation_directory, filename)
        rewrite_class_id(annotation_path, new_class_id)
