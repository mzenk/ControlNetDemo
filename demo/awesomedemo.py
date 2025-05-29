import random

import einops
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from image_utils import take_luminance_from_first_chroma_from_second
from pytorch_lightning import seed_everything

import config
from annotator.canny import CannyDetector
from annotator.util import HWC3, resize_image
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from cldm.model import create_model, load_state_dict


@torch.no_grad()
def process(
    model,
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
    img = resize_image(HWC3(input_image), image_resolution)
    H, W, C = img.shape

    apply_canny = CannyDetector()
    ddim_sampler = DDIMSampler(model)

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


def main():
    # Instead of "from share import *", execute the shared code directly
    disable_verbosity()

    if config.save_memory:
        enable_sliced_attention()

    model = create_model("./models/cldm_v15.yaml").cpu()
    model.load_state_dict(
        load_state_dict("./models/control_sd15_canny.pth", location="cuda")
    )
    model = model.cuda()

    # Load the PNG image into a numpy array
    input_image = imageio.imread("test_imgs//mri_brain.jpg")

    # Print the shape of the array
    print(input_image.shape)
    plt.imshow(input_image)
    plt.savefig("input_img.png")

    low_threshold = 50
    high_threshold = 100
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

    result = process(
        model,
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

    detected_map = result[0]
    samples = result[1:]
    plt.imshow(detected_map)
    plt.show()
    plt.savefig("detected_map.png")

    fig, axes = plt.subplots(1, len(samples), figsize=(len(samples) * 5, 5))
    for ax, s in zip(axes, samples):
        ax.imshow(s)
        ax.axis(False)
    fig.savefig("samples.png")

    index = -1
    test = take_luminance_from_first_chroma_from_second(
        resize_image(HWC3(input_image), image_resolution), samples[index], mode="lab"
    )

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(input_image)
    axs[1].imshow(samples[index])
    axs[2].imshow(test)

    axs[0].axis(False)
    axs[1].axis(False)
    axs[2].axis(False)

    plt.show()
    fig.savefig("fig2.png")


if __name__ == "__main__":
    main()
