from PIL import Image
import os
import shutil

# This file scales images from NIH chest X-ray and Vindr-CXR datasets.
# Images from these two datasets have large dimensions that affect the training speed if we scale during training.

def preprocess(src_dir, dest_dir, basesize=320):
    # Clear dest_dir
    try:
        shutil.rmtree(dest_dir)
    except:
        pass
    os.makedirs(dest_dir)
    file_lists = os.listdir(src_dir)
    for file in file_lists:
        if file.endswith(('.jpg', '.png', 'jpeg')):
            src_img_path = src_dir + file
            img = Image.open(src_img_path)
            (width, height) = img.size[-2:]
            scale = basesize/float(min(width, height))
            img = img.resize((max(basesize, int(width * scale)), max(basesize, int(height * scale))), Image.Resampling.LANCZOS)
            img.save(dest_dir + file)


if __name__ == '__main__':

    src_dir = "/slurm_data/rawdata/ChestX-ray14/images/"
    dest_dir = "/mnt/qb/work/baumgartner/sun22/data/NIH_data/images_rescaled_test/"
    preprocess(src_dir, dest_dir)
