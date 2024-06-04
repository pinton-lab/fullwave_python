import os
import random

import numpy as np
import scipy


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASSEED"] = str(seed)
    np.random.seed(seed)


def normalize_255(image):
    normalized_image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
    return normalized_image


def matlab_round(value):
    return np.round(value + 1e-9).astype(int)


def matlab_gaussian_filter(image, sigma):
    return scipy.ndimage.gaussian_filter(
        image.astype(float),
        sigma=sigma,
        radius=np.ceil(2 * sigma).astype(int),
    )


def matlab_interp2easy(image, interpolation_x, interpolation_y):
    dxi = 1 / matlab_round(image.shape[0] * interpolation_x - 1)
    xvec = np.arange(0, 1 + dxi, dxi)
    xvec = xvec * (image.shape[0] - 1)

    dyi = 1 / matlab_round(image.shape[1] * interpolation_y - 1)
    yvec = np.arange(0, 1 + dyi, dyi)
    yvec = yvec * (image.shape[1] - 1)
    xi, yi = np.meshgrid(xvec, yvec)

    image_out = (
        scipy.ndimage.map_coordinates(image, [xi.ravel(), yi.ravel()], order=0, mode="nearest")
        .reshape(yvec.shape[0], xvec.shape[0])
        .T
    )
    return image_out


def power_compress(image, power=1 / 4):
    sign = np.sign(image)
    image = np.power(np.abs(image), power)
    image = sign * image
    return image


def detect_envelope(image):
    return np.abs(scipy.signal.hilbert(image, axis=0))


def cut_domain(map_data, i_event, num_x, num_y, n_events, beam_spacing):
    orig = int(
        np.round(map_data.shape[0] / 2)
        - np.round(num_x / 2)
        - np.round(((i_event + 1) - (n_events + 1) / 2) * beam_spacing)
        - 6
    )
    map_data = map_data[orig : orig + num_x, 0:num_y]
    return map_data
