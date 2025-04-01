# script to download the masks images from the dataset from kaggle s direct downloading is not supporting it.

import shutil
import os
import zipfile

# Paths inside Kaggle
source_areas = "/kaggle/input/fs2020-runway-dataset/labels/labels/areas"
source_lines = "/kaggle/input/fs2020-runway-dataset/labels/labels/lines"

# Destination in working directory
dest_areas = "/kaggle/working/areas"
shutil.copytree(source_areas, dest_areas)


shutil.copy(f"{source_lines}/train_labels_640x360.json", "/kaggle/working/")
shutil.copy(f"{source_lines}/test_labels_640x360.json", "/kaggle/working/")

def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, os.path.dirname(folder_path))
                zipf.write(abs_path, arcname=rel_path)

zip_folder(dest_areas, "/kaggle/working/areas.zip")