"""Image pre-processing from Part 1: longest-edge resizing and whitespace cropping."""
import numpy as np
from PIL import Image


def resize_max_edge(img, max_edge=None):
    if max_edge and max(img.size) > max_edge:
        s = max_edge / float(max(img.size))
        img = img.resize((int(img.width * s), int(img.height * s)), Image.Resampling.LANCZOS)
    return img


def crop_whitespace(img, thresh=235, margin=10):
    """Crop near-white borders. Photos rarely have white borders, so they are mostly left unchanged."""
    g = np.asarray(img.convert("L"))
    rows, cols = np.where((g < thresh).any(1))[0], np.where((g < thresh).any(0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    box = (max(cols[0] - margin, 0), max(rows[0] - margin, 0),
           min(cols[-1] + margin + 1, img.width), min(rows[-1] + margin + 1, img.height))
    return img.crop(box)


def preprocess(img, max_edge=None, crop=False):
    if crop:
        img = crop_whitespace(img)
    return resize_max_edge(img, max_edge)
