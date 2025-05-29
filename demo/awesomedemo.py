import random

import einops
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from conditioning import get_canny_conditioning
from image_utils import take_luminance_from_first_chroma_from_second
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from annotator.util import HWC3, resize_image
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from cldm.model import create_model, load_state_dict


@torch.no_grad()
def generate_samples(
    model,
    cond_image,
    prompt,
    a_prompt,
    n_prompt,
    num_samples,
    ddim_steps,
    guess_mode,
    strength,
    scale,
    seed,
    eta,
    save_memory,
):
    """
    Generates image samples using a diffusion model with Canny edge control.

    Args:
        model: The diffusion model to use for image generation.
        cond_image (np.ndarray): The conditioning image (same shape as the input image).
        prompt (str): The main text prompt describing the desired output.
        a_prompt (str): Additional positive prompt to enhance the main prompt.
        n_prompt (str): Negative prompt to specify undesired features.
        num_samples (int): Number of samples to generate.
        ddim_steps (int): Number of DDIM sampling steps.
        guess_mode (bool): Whether to use guess mode for control scales.
        strength (float): Strength of the control signal.
        scale (float): Guidance scale for classifier-free guidance.
        seed (int): Random seed for reproducibility. If -1, a random seed is used.
        eta (float): DDIM eta parameter for stochasticity.
        save_memory (bool): Whether to enable memory-saving mode.

    Returns:
        List[np.ndarray]: A list containing the inverted Canny edge map followed by the generated image samples.
    """
    H, W, C = cond_image.shape
    ddim_sampler = DDIMSampler(model)

    control = torch.from_numpy(cond_image.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, "b h w c -> b c h w").clone()

    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    if save_memory:
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

    if save_memory:
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

    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (
        (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )

    return [x_samples[i] for i in range(num_samples)]


def main():
    cfg = OmegaConf.load("demo/config.yaml")

    # Instead of "from share import *", execute the shared code directly
    disable_verbosity()
    if cfg.save_memory:
        enable_sliced_attention()

    # Load model
    model = create_model(cfg.model_config).cpu()
    model.load_state_dict(load_state_dict(cfg.model_weights, location="cuda"))
    model = model.cuda()  # could make this configurable (CPU/GPU)

    # Get the conditioning image
    cond_image = get_canny_conditioning(
        cfg.input_image,
        image_resolution=cfg.image_resolution,
        low_threshold=cfg.conditioning.low_threshold,
        high_threshold=cfg.conditioning.high_threshold,
    )

    # Run
    samples = generate_samples(
        model=model,
        cond_image=cond_image,
        prompt=cfg.prompt,
        a_prompt=cfg.a_prompt,
        n_prompt=cfg.n_prompt,
        num_samples=cfg.num_samples,
        ddim_steps=cfg.ddim_steps,
        guess_mode=cfg.guess_mode,
        strength=cfg.strength,
        scale=cfg.scale,
        seed=cfg.seed,
        eta=cfg.eta,
        save_memory=cfg.save_memory,
    )

    # Show results
    index = -1
    input_image = imageio.imread(cfg.input_image)
    test = take_luminance_from_first_chroma_from_second(
        resize_image(HWC3(input_image), cfg.image_resolution),
        samples[index],
        mode="lab",
    )

    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    # It would be better to imshow resize_image(HWC3(input_image), cfg.image_resolution) instead of input_image
    # but it is not resized/RGB converted in the original code
    axs[0].imshow(input_image)
    axs[0].set_title("Input Image")
    axs[1].imshow(cond_image)
    axs[1].set_title("Canny Edge Map")
    axs[2].imshow(samples[index])
    axs[2].set_title("Generated Sample")
    axs[3].imshow(test)
    axs[3].set_title("Luminance from Sample")

    for ax in axs:
        ax.axis(False)

    plt.show()
    fig.savefig(cfg.output_image)
    print(f"Output saved to {cfg.output_image}")


if __name__ == "__main__":
    main()
