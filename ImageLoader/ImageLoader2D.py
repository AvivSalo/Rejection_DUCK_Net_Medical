import glob
import numpy as np
import os
from PIL import Image
from skimage.io import imread
from tqdm import tqdm

def load_data(img_height, img_width, images_to_be_loaded, dataset, data_path, split='train'):
    IMAGES_PATH = os.path.join(data_path, split, 'images/')
    MASKS_PATH = os.path.join(data_path, split, 'masks/')

    if dataset == 'kvasir':
        train_ids = glob.glob(IMAGES_PATH + "*.jpg")
    elif dataset == 'cvc-clinicdb':
        train_ids = glob.glob(IMAGES_PATH + "*.tif")
    elif dataset in ['cvc-colondb', 'etis-laribpolypdb']:
        train_ids = glob.glob(IMAGES_PATH + "*.png")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if not train_ids:
        raise ValueError(f"No images found in {IMAGES_PATH}. Check the folder structure.")

    if images_to_be_loaded == -1:
        images_to_be_loaded = len(train_ids)

    X_data = np.zeros((images_to_be_loaded, img_height, img_width, 3), dtype=np.float32)
    Y_data = np.zeros((images_to_be_loaded, img_height, img_width), dtype=np.uint8)

    print(f'Resizing {split} images and masks: {images_to_be_loaded}')
    for n, id_ in tqdm(enumerate(train_ids)):
        if n == images_to_be_loaded:
            break

        image_path = id_
        mask_path = image_path.replace("images", "masks")

        image = imread(image_path)
        mask_ = imread(mask_path)

        pillow_image = Image.fromarray(image)
        pillow_image = pillow_image.resize((img_height, img_width))
        image = np.array(pillow_image)

        X_data[n] = image / 255

        pillow_mask = Image.fromarray(mask_)
        pillow_mask = pillow_mask.resize((img_height, img_width), resample=Image.LANCZOS)
        mask_ = np.array(pillow_mask)

        # Convert to binary (assuming mask_ is RGB or grayscale)
        binary_mask = (mask_ > 127).astype(np.uint8)
        if len(binary_mask.shape) == 3:  # If RGB
            binary_mask_single_channel = np.max(binary_mask, axis=-1)
        else:  # If grayscale
            binary_mask_single_channel = binary_mask

        Y_data[n] = binary_mask_single_channel

    Y_data = np.expand_dims(Y_data, axis=-1)

    return X_data, Y_data