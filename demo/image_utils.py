import cv2
import numpy as np


# image processing functions
def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


def lab2rgb(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def rgb2yuv(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)


def yuv2rgb(yuv: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


def srgb2lin(s):
    s = s.astype(float) / 255.0
    return np.where(
        s <= 0.0404482362771082, s / 12.92, np.power(((s + 0.055) / 1.055), 2.4)
    )


def lin2srgb(lin):
    return 255 * np.where(
        lin > 0.0031308, 1.055 * (np.power(lin, (1.0 / 2.4))) - 0.055, 12.92 * lin
    )


def get_luminance(
    linear_image: np.ndarray, luminance_conversion=[0.2126, 0.7152, 0.0722]
):
    return np.sum([[luminance_conversion]] * linear_image, axis=2)


def take_luminance_from_first_chroma_from_second(luminance, chroma, mode="lab", s=1):
    assert luminance.shape == chroma.shape, f"{luminance.shape=} != {chroma.shape=}"
    if mode == "lab":
        lab = rgb2lab(chroma)
        lab[:, :, 0] = rgb2lab(luminance)[:, :, 0]
        return lab2rgb(lab)
    if mode == "yuv":
        yuv = rgb2yuv(chroma)
        yuv[:, :, 0] = rgb2yuv(luminance)[:, :, 0]
        return yuv2rgb(yuv)
    if mode == "luminance":
        lluminance = srgb2lin(luminance)
        lchroma = srgb2lin(chroma)
        return lin2srgb(
            np.clip(
                lchroma
                * ((get_luminance(lluminance) / (get_luminance(lchroma))) ** s)[
                    :, :, np.newaxis
                ],
                0,
                1,
            )
        )
