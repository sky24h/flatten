import cv2
import time
import torch
import imageio
import numpy as np
from PIL import Image

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDIMScheduler, AutoencoderKL, DDIMInverseScheduler

from models.pipeline_flatten import FlattenPipeline
from models.util import sample_trajectories
from models.unet import UNet3DConditionModel


def init_pipeline(device, sd_path="checkpoints/stable-diffusion-2-1-base"):
    dtype        = torch.float16
    unet         = UNet3DConditionModel.from_pretrained_2d(sd_path, subfolder="unet").to(dtype=torch.float16)
    vae          = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(dtype=torch.float16)
    tokenizer    = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer", dtype=dtype)
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(dtype=torch.float16)
    scheduler    = DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler")
    inverse      = DDIMInverseScheduler.from_pretrained(sd_path, subfolder="scheduler")

    pipe = FlattenPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, inverse_scheduler=inverse)
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to(device)
    return pipe


height       = 512
width        = 512
sample_steps = 50
inject_step  = 40

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
pipe = init_pipeline(device)


def inference(
    seed          : int = 66,
    prompt        : str = None,
    neg_prompt    : str = "",
    guidance_scale: float = 10.0,
    video_length  : int = 32,
    video_path    : str = None,
    output_dir    : str = None,
    frame_rate    : int = 1,
    fps           : int = 15,
    old_qk        : int = 0,
):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # read the source video
    video_reader = imageio.get_reader(video_path, "ffmpeg")
    video = []
    for frame in video_reader:
        if len(video) >= video_length:
            break
        video.append(cv2.resize(frame, (width, height)))  # .transpose(2, 0, 1))
    real_frames = [Image.fromarray(frame) for frame in video]

    # compute optical flows and sample trajectories
    trajectories = sample_trajectories(torch.tensor(np.array(video)).permute(0, 3, 1, 2), device)
    torch.cuda.empty_cache()

    for k in trajectories.keys():
        trajectories[k] = trajectories[k].to(device)
    sample = (
        pipe(
            prompt,
            video_length        = video_length,
            frames              = real_frames,
            num_inference_steps = sample_steps,
            generator           = generator,
            guidance_scale      = guidance_scale,
            negative_prompt     = neg_prompt,
            width               = width,
            height              = height,
            trajs               = trajectories,
            output_dir          = "tmp/",
            inject_step         = inject_step,
            old_qk              = old_qk,
        )
        .videos[0]
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
        * 255
    ).astype(np.uint8)
    temp_video_name = f"/tmp/{prompt}_{neg_prompt}_{str(guidance_scale)}_{time.time()}.mp4".replace(" ", "-")
    video_writer = imageio.get_writer(temp_video_name, fps=fps)
    for frame in sample:
        video_writer.append_data(frame)
    print(f"Saving video to {temp_video_name}, sample shape: {sample.shape}")
    return temp_video_name


if __name__ == "__main__":
    video_path = "./data/puff.mp4"
    generated_video = inference(
        video_path     = video_path,
        prompt         = "A Tiger, high quality",
        neg_prompt     = "a cat with big eyes, deformed",
        guidance_scale = 20,
        old_qk         = 0,
    )
