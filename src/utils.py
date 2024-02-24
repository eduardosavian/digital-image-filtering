import os

import cv2 as cv
import numpy as np

def GetDirectory():
    current_path = os.getcwd()

    path_main =  "/".join(current_path.split("/")[:-1])

    return path_main

def read_image(name: str, itype: str) -> np.ndarray:
    path = os.path.join(GetDirectory(), "images", itype, name)
    #print(f"Read on: {path}")

    if not os.path.isfile(path):
        print(f"Image file '{path}' does not exist.")
        return

    image = cv.imread(path, cv.IMREAD_UNCHANGED)

    return image

def write_plot_image(name: str, itype: str, plt_image):
    path = os.path.join(GetDirectory(), "images","manipulations", itype, name)

    path = path.replace('.tif', '') + '.png'
    #print(f"Write on: {path}")

    plt_image.savefig(path, dpi=300)

def write_image(name: str, itype: str, image: np.ndarray):
    path = os.path.join(GetDirectory(), "images","manipulations", itype, name)
    #print(f"Write on: {path}")

    cv.imwrite(path, image)


def is_monochrome(image_path: str) -> bool:
    path = os.path.join(GetDirectory(), "images", "repository", image_path)
    img = cv.imread(path, cv.IMREAD_UNCHANGED)
    return img.shape[2] == 1


def normalize_image(img: np.ndarray, depth: int = 8) -> np.ndarray:
    N = np.float32((2 ** depth) - 1)

    def normalize(n: np.float32):
        return n / N

    normalizev = np.vectorize(normalize)

    norm = img.astype(np.float32)
    norm = normalizev(norm)

    return norm


def denormalize_image(norm, depth=8):
    max_value = np.float32((2 ** depth) - 1)

    denormalizev = np.vectorize(lambda n: np.floor(n * max_value))

    denorm = denormalizev(norm)

    if depth == 8:
        denorm = denorm.astype(np.uint8)
    elif depth == 16:
        denorm = denorm.astype(np.uint16)
    elif depth == 32:
        denorm = denorm.astype(np.uint32)
    elif depth <= 8:
        if depth == 0: max_value = np.float32(1)
        frac = 255 / max_value
        denorm = (denorm * frac).astype(np.uint8)
    else:
        assert False, "Unsupported depth"

    return denorm

