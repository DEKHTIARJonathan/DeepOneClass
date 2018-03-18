import os
import numpy as np
from glob import glob
from image_ops import get_image


def get_train_val_imgs(class_nbr, data_dir, target_w):
    data_path = os.path.join(data_dir, "Class{}".format(class_nbr))
    data_path_files = glob(os.path.join(data_path, "*.png"))
    np.random.shuffle(data_path_files)

    data_def_path = os.path.join(data_dir, "Class{}_def".format(class_nbr))
    data_def_path_files = glob(os.path.join(data_def_path, "*.png"))
    np.random.shuffle(data_def_path_files)

    num_test_def = len(data_def_path_files)
    num_train = len(data_path_files) - num_test_def

    train_images = np.array(list(map(lambda f: get_image(f, target_w=target_w), data_path_files[:num_train])))
    train_images = np.stack([train_images]*3, axis=3).squeeze(4)
    assert (train_images.shape == (num_train, target_w, target_w, 3))


    test_images = np.array(list(map(lambda f: get_image(f, target_w=target_w), data_path_files[num_train:] + data_def_path_files)))
    test_images = np.stack([test_images]*3, axis=3).squeeze(4)
    assert (test_images.shape == (num_test_def*2, target_w, target_w, 3))
    test_images_y = np.concatenate((np.ones(len(data_path_files) - num_train), -1 * np.ones(len(data_def_path_files))), axis=0)

    return train_images, test_images, test_images_y