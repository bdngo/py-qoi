"""
Module for encoding images to the QOI format.
"""
from typing import Union

import bitarray as ba
import numpy as np
from bitarray.util import int2ba

from utils import px_hash


def qoi_op_rgb(px: np.ndarray) -> bytes:
    """
    Returns the QOI_OP_RGB pixel encoding.
    """
    return b"\xfe" + px[:3].tobytes()


def qoi_op_rgba(px: np.ndarray) -> bytes:
    """
    Returns the QOI_OP_RGBA pixel encoding.
    """
    return b"\xff" + px.tobytes()


def qoi_op_index(idx: int) -> bytes:
    """
    Returns the QOI_OP_INDEX pixel encoding.
    """
    return ba.bitarray("00") + int2ba(idx, length=6)


def qoi_op_diff(px_diffs: np.ndarray) -> bytes:
    """
    Returns the QOI_OP_DIFF pixel encoding.
    """
    px_diffs_biased = px_diffs + 2
    return (ba.bitarray("01") \
        + int2ba(px_diffs_biased[0].item(), length=2) \
        + int2ba(px_diffs_biased[1].item(), length=2) \
        + int2ba(px_diffs_biased[2].item(), length=2)).tobytes()


def qoi_op_luma(px_diffs: np.ndarray) -> bytes:
    """
    Returns the QOI_OP_LUMA pixel encoding.
    """
    return (ba.bitarray("10") \
        + int2ba(px_diffs[0].item() + 32, length=6) \
        + int2ba(px_diffs[1].item() + 8, length=4) \
        + int2ba(px_diffs[2].item() + 8, length=4)).tobytes()


def qoi_op_run(run: int) -> bytes:
    """
    Returns the QOI_OP_RUN pixel encoding.
    """
    return (ba.bitarray("11") + int2ba(run - 1, length=6)).tobytes()


def unsigned_sub(px1: Union[np.ndarray, int], px2: Union[np.ndarray, int]) -> int:
    """
    Performs subtraction between unsigned integers, then casts as signed integers.
    """
    return np.byte(np.ubyte(px1) - np.ubyte(px2))


def encode(img: np.ndarray) -> bytes:
    """
    Encodes a Numpy array of pixels into the QOI image format.
    """
    height, width, chans = img.shape
    num_px = width * height
    img_flat = img.reshape(num_px, chans)

    px_arr = np.zeros((64, chans))
    px_prev = np.array([0, 0, 0, 255]) if chans == 4 else np.array([0, 0, 0])
    img_compressed = []
    run = 0
    for ptr in range(num_px):
        px = img_flat[ptr]
        if np.all(np.equal(px, px_prev)):
            run += 1
            if run == 62 or ptr == num_px - 1:
                img_compressed.append(qoi_op_run(run))
                run = 0
        else:
            if run > 0:
                img_compressed.append(qoi_op_run(run))
                run = 0
            idx = px_hash(px.tolist())
            if np.all(np.equal(px_arr[idx], px)):
                img_compressed.append(qoi_op_index(idx))
            else:
                px_arr[idx] = px

                if px[3] == px_prev[3]:
                    px_diff = np.apply_along_axis(unsigned_sub, px[:3], px_prev[:3])
                    drdg = unsigned_sub(px_diff[0], px_diff[1])
                    dbdg = unsigned_sub(px_diff[2], px_diff[1])
                    if ((-2 <= px_diff) & (px_diff < 2)).all():
                        img_compressed.append(qoi_op_diff(px_diff))
                    elif -32 <= px_diff[1] < 32 and -8 <= drdg < 8 and -8 <= dbdg < 8:
                        img_compressed.append(qoi_op_luma(np.array([px_diff[1], drdg, dbdg])))
                    else:
                        img_compressed.append(qoi_op_rgb(px))
                else:
                    img_compressed.append(qoi_op_rgba(px))
        px_prev = px

    header = b"qoif" \
        + width.to_bytes(4, "big") \
        + height.to_bytes(4, "big") \
        + chans.to_bytes(1, "big") + b"\x00"
    tail = b"\x00" * 7 + b"\x01"

    return header + b''.join(img_compressed) + tail
