import numpy as np
import os
from PIL import Image


def color_to_binary_mask(img_path, mask_color=(255, 0, 0), save_path=None):
    """
    Create a binary (black/white) mask from an input image containing a particular color. The default color is red.
    Returns a PIL image.
    """
    img = Image.open(img_path).convert("RGB")  # get rid of alpha channel
    img_arr = np.asarray(img)
    mask = np.all(img_arr == mask_color, axis=-1)
    if not np.any(mask):
        print("skipping: mask_color not found")
        return None

    new_img_arr = np.zeros(mask.shape, dtype=np.uint8)  # black
    new_img_arr[mask] = 255  # white
    binary_mask = Image.fromarray(new_img_arr, "L")  # 1 channel grayscale
    if save_path:
        if os.path.splitext(save_path)[-1].lower() != ".png":
            print("warning: a lossless format such as PNG is recommended")
        binary_mask.save(save_path)
    return binary_mask


def swap_colors(img_path, color_a=(0, 0, 0), color_b=(0, 0, 0), save_path=None):
    """
    Swaps all pixels of one color with another color and vice versa. Used to fix accidental annotation color mistakes!
    """
    img = Image.open(img_path).convert("RGB")  # get rid of alpha channel
    img_arr = np.asarray(img).copy()
    color_a_mask = np.all(img_arr == color_a, axis=-1)
    color_b_mask = np.all(img_arr == color_b, axis=-1)
    img_arr[color_a_mask] = color_b
    img_arr[color_b_mask] = color_a
    new_img = Image.fromarray(img_arr)
    new_img.save(save_path)
