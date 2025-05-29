import random

import cv2
import einops
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import seed_everything

import config
from annotator.canny import CannyDetector
from annotator.util import HWC3, resize_image
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from cldm.model import create_model, load_state_dict

# Instead of "from share import *", execute the shared code directly
disable_verbosity()

if config.save_memory:
    enable_sliced_attention()


apply_canny = CannyDetector()

model = create_model("./models/cldm_v15.yaml").cpu()
model.load_state_dict(
    load_state_dict("./models/control_sd15_canny.pth", location="cuda")
)
model = model.cuda()
ddim_sampler = DDIMSampler(model)


# %%
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


# %%
# utils functions


def process(
    input_image,
    prompt,
    a_prompt,
    n_prompt,
    num_samples,
    image_resolution,
    ddim_steps,
    guess_mode,
    strength,
    scale,
    seed,
    eta,
    low_threshold,
    high_threshold,
):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {
            "c_concat": [control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
            ],
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)]
            if guess_mode
            else ([strength] * 13)
        )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results


# %%
# Load the PNG image into a numpy array
input_image = imageio.imread("test_imgs//mri_brain.jpg")

# Print the shape of the array
print(input_image.shape)

plt.imshow(input_image)

# %%
low_threshold = 50
high_threshold = 100
detected_map = apply_canny(input_image, low_threshold, high_threshold)
detected_map = HWC3(detected_map)

# detected_map = feature.canny(rgb2gray(input_image), sigma=2).astype(np.float32)
# detected_map = filters.roberts(rgb2gray(input_image))
# detected_map = filters.sobel(rgb2gray(input_image))
# detected_map = filters.hessian(rgb2gray(input_image), range(1, 10))

# detected_map = np.clip(detected_map.astype(np.float32) * 255, 0, 255).astype(np.uint8)
# detected_map = HWC3(detected_map)

plt.imshow(255 - detected_map)
# plt.imshow(detected_map)
plt.show()


# %%

prompt = "mri brain scan"
num_samples = 1
image_resolution = 512
strength = 1.0
guess_mode = False
low_threshold = 50
high_threshold = 100
ddim_steps = 10
scale = 9.0
seed = 1
eta = 0.0
a_prompt = "good quality"  # 'best quality, extremely detailed'
n_prompt = "animal, drawing, painting, vivid colors, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"


ips = [
    input_image,
    prompt,
    a_prompt,
    n_prompt,
    num_samples,
    image_resolution,
    ddim_steps,
    guess_mode,
    strength,
    scale,
    seed,
    eta,
    low_threshold,
    high_threshold,
]


# %%

result = process(
    input_image=input_image,
    prompt=prompt,
    a_prompt=a_prompt,
    n_prompt=n_prompt,
    num_samples=num_samples,
    image_resolution=image_resolution,
    ddim_steps=ddim_steps,
    guess_mode=guess_mode,
    strength=strength,
    scale=scale,
    seed=seed,
    eta=eta,
    low_threshold=low_threshold,
    high_threshold=high_threshold,
)

for res in result:
    plt.imshow(res)
    plt.axis(False)
    plt.show()

# %%
index = -1
test = take_luminance_from_first_chroma_from_second(
    resize_image(HWC3(input_image), image_resolution), result[index], mode="lab"
)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(input_image)
axs[1].imshow(result[index])
axs[2].imshow(test)

axs[0].axis(False)
axs[1].axis(False)
axs[2].axis(False)

plt.show()
# %%
