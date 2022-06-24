"""
Decoder for QOI images.
"""
import io

import bitarray as ba
import numpy as np
from bitarray.util import ba2int
from PIL import Image

from utils import px_hash


def decode(img_comp: bytes) -> Image:
    """
    Decodes a QOI image.
    """
    img_stream = io.BytesIO(img_comp)
    if img_stream.read(4) != b"qoif":
        raise IOError("Not a valid QOI file")

    width = int.from_bytes(img_stream.read(4), "big")
    height = int.from_bytes(img_stream.read(4), "big")
    chans = int.from_bytes(img_stream.read(1), "big")
    cspace = int.from_bytes(img_stream.read(1), "big")
    if width < 0 or height < 0 or chans not in (3, 4) or cspace > 1:
        raise IOError("Bad file metadata")

    px_cache = np.zeros((64, chans), dtype=np.uint8)
    img_arr = []
    px_prev = np.array([0, 0, 0, 255]) if chans == 4 else np.array([0, 0, 0])
    px = px_prev
    while len(img_arr) < width * height:
        chunk = img_stream.read(1)
        if chunk == b"\xfe":
            px_r = int.from_bytes(img_stream.read(1), "big")
            px_g = int.from_bytes(img_stream.read(1), "big")
            px_b = int.from_bytes(img_stream.read(1), "big")
            px = np.array([px_r, px_g, px_b, px_prev[-1]])
        elif chunk == b"\xff":
            px_r = int.from_bytes(img_stream.read(1), "big")
            px_g = int.from_bytes(img_stream.read(1), "big")
            px_b = int.from_bytes(img_stream.read(1), "big")
            px_a = int.from_bytes(img_stream.read(1), "big")
            px = np.array([px_r, px_g, px_b, px_a])
        else:
            chunk_ba = ba.bitarray()
            chunk_ba.frombytes(chunk)
            chunk_header = chunk_ba[:2]
            chunk_data = chunk_ba[2:]

            if chunk_header == ba.bitarray("00"):
                px = px_cache[ba2int(chunk_data)]
            elif chunk_header == ba.bitarray("01"):
                diff_r = ba2int(chunk_data[:2]) - 2
                diff_g = ba2int(chunk_data[2:4]) - 2
                diff_b = ba2int(chunk_data[4:]) - 2
                if chans == 3:
                    px = (px_prev + np.array([diff_r, diff_g, diff_b])) % 256
                elif chans == 4:
                    px = (px_prev + np.array([diff_r, diff_g, diff_b, 0])) % 256
            elif chunk_header == ba.bitarray("10"):
                luma = ba.bitarray()
                luma.frombytes(img_stream.read(1))
                diff_g = ba2int(chunk_data) - 32
                drdg = ba2int(luma[:4]) - 8
                dbdg = ba2int(luma[4:]) - 8
                if chans == 3:
                    px = (px_prev + np.array([diff_g + drdg, diff_g, diff_g + dbdg])) % 256
                elif chans == 4:
                    px = (px_prev + np.array([diff_g + drdg, diff_g, diff_g + dbdg, 0])) % 256
            elif chunk_header == ba.bitarray("11"):
                num_runs = ba2int(chunk_data)
                img_arr.extend([px_prev] * num_runs)
                continue

        idx = px_hash(px.tolist())
        px_cache[idx] = px
        img_arr.append(px)
        px_prev = px

    tail = img_stream.read(8)
    if tail != b"\x00" * 7 + b"\x01":
        raise IOError("Invalid QOI tail")

    return Image.fromarray(np.array(img_arr).reshape((height, width, chans)).astype(np.uint8))


def main() -> None:
    PATH = "/home/bngo/Programming/py-qoi/test/qoi_test_images/testcard.qoi"
    with open(PATH, 'rb') as f:
        img = decode(f.read())
        img.save("testcard.png")


if __name__ == "__main__":
    main()
