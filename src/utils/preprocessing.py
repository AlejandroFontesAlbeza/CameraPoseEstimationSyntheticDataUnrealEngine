from PIL import Image
import numpy as np
import os

from utils.dict_utils import exact_color_palette, range_color_palette



def data_process(input_mask_path, output_mask_path, input_image_path, output_image_path):
    """
    Procesa una máscara y una imagen:
    - Convierte la máscara a escala de grises según la paleta y la reescala a 512x512.
    - Reescala la imagen a 512x512.
    """
    mask = Image.open(input_mask_path).convert("RGB")
    mask_pixels = np.array(mask)
    height, width = mask_pixels.shape[:2]
    gray_mask = np.zeros((height, width), dtype=np.uint8)

    r_channel = mask_pixels[:, :, 0]
    g_channel = mask_pixels[:, :, 1]
    b_channel = mask_pixels[:, :, 2]

    # Exact color matches
    for color, class_index in exact_color_palette.items():
        match = np.all(mask_pixels == color, axis=-1)  # shape (H, W)
        gray_mask[match] = class_index

    # Range-based matches
    for class_info in range_color_palette.values():
        (r_min, r_max), (g_min, g_max), (b_min, b_max) = class_info["range"]
        class_index = class_info["class_index"]
        match = (
            (r_channel >= r_min) & (r_channel <= r_max) &
            (g_channel >= g_min) & (g_channel <= g_max) &
            (b_channel >= b_min) & (b_channel <= b_max)
        )
        gray_mask[match] = class_index
    gray_mask_image = Image.fromarray(gray_mask)
    gray_mask_image_resized = gray_mask_image.resize((512,512), resample=Image.NEAREST)
    gray_mask_image_resized.save(output_mask_path)

    image = Image.open(input_image_path).convert("RGB")
    image_resized = image.resize((512,512), resample=Image.BILINEAR)
    image_resized.save(output_image_path)


def verify_split(images_folder, masks_folder):
    image_files = sorted(os.listdir(images_folder))
    mask_files = sorted(os.listdir(masks_folder))

    image_set = set(image_files)
    mask_set = set(mask_files)

    missing_masks = image_set - mask_set
    missing_images = mask_set - image_set

    if len(missing_masks) == 0 and len(missing_images) == 0:
        print(f"Correct Verification in {images_folder}")
        print(f"Total files: {len(image_files)}")
    else:
        print("Problem detected")
        if missing_masks:
            print("Images without mask:", missing_masks)
        if missing_images:
            print("Masks without image:", missing_images)
