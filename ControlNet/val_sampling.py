from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os
from PIL import Image
import argparse

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from vid_helper import ctc_2_track
from config_loader import load_config


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint file")
parser.add_argument("--prompt", type=str, required=True, help="Prompt for the model")
parser.add_argument("--epoch", type=int, required=True, help="Current epoch number")
parser.add_argument("--gpu", type=int, required=True, help="gpu")
args = parser.parse_args()


conf_p = os.environ.get('CONFIG_PATH')
conf = load_config(conf_p)
name = conf["name"]

ckpt_path = args.ckpt_path
prompt = args.prompt
epoch = args.epoch
gpu=args.gpu

device =torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu") #torch.device("cpu") #torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
model = create_model('./models/cldm_v15.yaml').to(device)
if os.path.exists(ckpt_path):
    model.load_state_dict(load_state_dict(ckpt_path, location='cpu'))
else:
    print(f"Checkpoint file not found at {ckpt_path}. Loading default checkpoint.")
    model.load_state_dict(load_state_dict('./models/control_sd15_cell.ckpt', location='cpu'))
model = model.to(device)
ddim_sampler = DDIMSampler(model)
#torch.no_grad()
#torch.cuda.empty_cache()

def make_val_pic(prompt,epoch):
    input_dir = f'./sampling/val/id/{name}' # Replace this with your input images directory
    output_dir = f'./sampling/val/{name}/{epoch}'  # Output directory now contains the epoch number
    os.makedirs(output_dir, exist_ok=True)
    a_prompt =''
    n_prompt = ''
    num_samples = 1
    image_resolution = 512
    ddim_steps = 50
    guess_mode = False
    strength = 1.0
    scale = 9.0
    seed = -1 #1454547764
    eta = 0.0
    low_threshold = 100
    high_threshold = 200


    print('input_dir!!!!',input_dir,os.listdir(input_dir))

    # Load all JPG images from the input directory
    image_paths = [os.path.join(input_dir, img) for img in os.listdir(input_dir) if img.endswith(".png")]
    print('imagepath!!!!',image_paths)
    for image_path in image_paths:
        input_image = np.array(Image.open(image_path))  # Read image using Pillow

        result = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold)[1]

        
        output_img = Image.fromarray(result.astype('uint8'))  # Convert result to PIL Image
        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + f'.png')
        print('saving', image_path, output_path )
        output_img.save(output_path)  # Save the image to the output directory







def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape
        #print('ips',prompt, a_prompt, n_prompt)

        detected_map = img
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results
















def main():
    # Assume other parameters are already defined
    make_val_pic(prompt,epoch)

if __name__ == "__main__":
    main()