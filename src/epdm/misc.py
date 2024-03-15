import math
import random
from functools import wraps
import numpy as np
import pickle
from PIL import Image

import torch


def fix_random_state(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def exists(x):
    return x is not None


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(t, length=1):
    if isinstance(t, tuple):
        return t
    return (t,) * length


def divisible_by(numer, denom):
    return (numer % denom) == 0


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def pkl_load(file_path):
    with open(file_path, "rb") as fr:
        data = pickle.load(fr, encoding="bytes")
    return data


def read_img(file_path):
    return Image.open(file_path)


def dump_img(data, file_path):
    if isinstance(data, np.ndarray):
        image = Image.fromarray(data)
    image.save(file_path)
