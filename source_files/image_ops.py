"""
From https://github.com/DEKHTIARJonathan/AnoGAN/blob/master/image_ops.py
"""
import scipy.misc
import numpy as np

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w])

def squash_image(x, target_h=64, target_w=64):
    # interp = ["nearest", "lanczos", "bilinear", "bicubic", "cubic"]
    return scipy.misc.imresize(x, [target_h, target_w], interp="nearest")

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float32)
    else:
        return scipy.misc.imread(path).astype(np.float).astype(np.float32)

def transform(image, target_w=64, crop_size=64, is_crop=False):
    if is_crop:
        cropped_image = center_crop(image, crop_size, resize_w=target_w)
    else:
        cropped_image = squash_image(image, target_h=target_w, target_w=target_w)
    return np.array(cropped_image) - 127.5

def get_image(image_path, target_w=64, is_grayscale = True, crop_size=64, is_crop=False):
    image = imread(image_path, is_grayscale)
    image = transform(image, target_w=target_w, crop_size=crop_size, is_crop=is_crop)

    if len(image.shape) < 3:
        return np.expand_dims(image, 3)
    else:
        return image