import os
import glob
import secrets
import sys


def rename_files(base_dir):
    for category in ['train', 'valid', 'test']:
        # Store the generated random strings in a dictionary to ensure
        # corresponding images and labels get the same suffix.
        random_strings = {}

        for filetype in ['images', 'labels']:
            folder_path = os.path.join(base_dir, category, filetype)

            if not os.path.exists(folder_path):
                print(f"Folder not found: {folder_path}")
                continue

            os.chdir(folder_path)
            ext = '.png' if filetype == 'images' else '.txt'
            files = glob.glob(f"*{ext}")

            for file in files:
                file_basename = os.path.splitext(file)[0]

                if file_basename not in random_strings:
                    random_strings[file_basename] = secrets.token_hex(3)

                random_suffix = random_strings[file_basename]
                _, file_ext = os.path.splitext(file)
                new_filename = f"{file_basename}_{random_suffix}{file_ext}"
                os.rename(file, new_filename)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python rename_script.py <dir1> <dir2> ...")
        sys.exit(1)

    for directory in sys.argv[1:]:
        if not os.path.isdir(directory):
            print(f"Invalid directory: {directory}")
            continue

        rename_files(directory)
        print(f"Processed directory: {directory}")
