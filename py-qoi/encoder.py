import bitarray as ba
import numpy as np
from bitarray.util import int2ba
from PIL import Image


def px_hash(px: list[int]) -> int:
    r, g, b = px[:3]
    a = px[3] if len(px) == 4 else 0
    return ((r * 3 + g * 5 + b * 7 + a * 11) % 64)


def qoi_op_rgb(px: np.ndarray) -> bytes:
    return b"\xff" + px.tobytes()


def qoi_op_index(idx: int) -> bytes:
    return ba.bitarray("00") + int2ba(idx, length=6)


def qoi_op_diff(px_diffs: np.ndarray) -> bytes:
    px_diffs_biased = px_diffs + 2
    return (ba.bitarray("01") \
        + int2ba(px_diffs_biased[0].astype(int), length=2) \
        + int2ba(px_diffs_biased[1].astype(int), length=2) \
        + int2ba(px_diffs_biased[2].astype(int), length=2)).tobytes()


def qoi_op_luma(px_diffs: np.ndarray) -> bytes:
    return (ba.bitarray("10") \
        + int2ba(px_diffs[0].astype(int) + 32, length=6) \
        + int2ba(px_diffs[1].astype(int) + 8, length=4) \
        + int2ba(px_diffs[2].astype(int) + 8, length=4)).tobytes()


def qoi_op_run(run: int) -> bytes:
    return (ba.bitarray("11") + int2ba(run - 1, length=6)).tobytes()


def encode(img: np.ndarray) -> bytes:
    num_px = img.shape[0] * img.shape[1]
    img_flat = img.reshape(num_px, img.shape[2])

    px_arr = np.zeros((64, img.shape[2]))
    px_prev = np.array([0, 0, 0, 255])
    img_compressed = []
    run = 0
    for ptr in range(num_px):
        px = img_flat[ptr]
        idx = px_hash(px.tolist())
        if np.all(np.equal(px, px_prev)):
            run += 1
            if run >= 62 or ptr <= num_px - 1:
                img_compressed.append(qoi_op_run(run))
                run = 0
        else:
            px_arr[idx] = px

            if run > 0:
                img_compressed.append(qoi_op_run(run))
                run = 0
            if np.all(np.equal(px_arr[idx], px)):
                img_compressed.append(qoi_op_index(idx))
            else:
                if px[3] == px_prev[3]:
                    px_diff = px[:3] - px_prev[:3]
                    drdg = px_diff[0] - px_diff[1]
                    dbdg = px_diff[2] - px_diff[1]
                    if ((-2 <= px_diff) & (px_diff < 2)).all():
                        img_compressed.append(qoi_op_diff(px_diff))
                    elif -32 <= px_diff[1] < 32 and -8 <= drdg < 8 and -8 <= dbdg < 8:
                        img_compressed.append(qoi_op_luma(np.array([px_diff[1], drdg, dbdg])))
                    else:
                        img_compressed.append(qoi_op_rgb(px))
                else:
                    img_compressed.append(qoi_op_rgb(px))
        px_prev = px
    
    width, height, chans = img.shape
    header = b"qoif" + width.to_bytes(4, "big") + height.to_bytes(4, "big") + chans.to_bytes(4, "big") + b"\x01"
    tail = b"\x00" * 7 + b"\x01"

    return header + b''.join(img_compressed) + tail


def main() -> None:
    pass


if __name__ == "__main__":
    main()

