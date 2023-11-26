import sys
import os
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)



from share import *
import config


import einops
import numpy as np
import torch
import random
from PIL import Image


from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from vid_helper import ctc_2_track, delete_files_in_folder






def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, model, ddim_sampler):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape
        print('ips',prompt, a_prompt, n_prompt)

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

def make_init_pic(name, device, input_dir, output_dir, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    
    resume_path = f'./ControlNet/models/{name}/last.ckpt'


    
    torch.cuda.set_device(device)


    model = create_model('./ControlNet/models/cldm_v15.yaml').to(device)
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model = model.to(device)
    ddim_sampler = DDIMSampler(model)
    
    
    # Get all JPG images from the input directory
    image_paths = [os.path.join(input_dir, img) for img in os.listdir(input_dir) if img.endswith(".jpg")]

    # Sort image paths based on the integer in the file name
    image_paths = sorted(image_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # Get the image with the highest number
    image_path = image_paths[-1]

    input_image = np.array(Image.open(image_path))  # Read image using Pillow

    result = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, model, ddim_sampler)[1]

    output_img = Image.fromarray(result.astype('uint8'))  # Convert result to PIL Image
    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + f'.png')
    #print('processing', image_path, output_path )
    output_img.save(output_path)  # Save the image to the output directory

    img_number = int(os.path.splitext(os.path.basename(image_path))[0])

    return img_number

def make_vid(name,device, num, id_path, res_path, cond_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    resume_path = f'./ControlNet/models/{name}_track/last.ckpt'
    A,pix=get_A_pix(name)
    #print('A,pix',A,pix)
    r_a=int(np.sqrt(A/ np.pi) / 4) if int(np.sqrt(A/ np.pi) / 4) > 2 else 2
    f=((pix)/(512))
    r_a = int(r_a /f)

    
    torch.cuda.set_device(device)

    model = create_model('./ControlNet/models/cldm_v15.yaml').to(device)
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model = model.to(device)
    ddim_sampler = DDIMSampler(model)

    for i in range(num, 0, -1):

        ctc_2_track(i, res_path, id_path, cond_path,r_a)
        image_paths = [os.path.join(cond_path, img) for img in os.listdir(cond_path) if img.endswith(".jpg")]

        # Sort image paths based on the integer in the file name
        image_paths = sorted(image_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        # Get the image with the lowest number
        image_path = image_paths[0]

        input_image = np.array(Image.open(image_path))  # Read image using Pillow # Read image using Pillow

        result = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, model, ddim_sampler)[1]

        output_img = Image.fromarray(result.astype('uint8'))  # Convert result to PIL Image
        output_file_num = int(os.path.splitext(os.path.basename(image_path))[0]) - 1
        output_path = os.path.join(res_path, str(output_file_num) + f'.png')
        #print('processing', image_path, output_path )
        output_img.save(output_path)  # Save the image to the output directory

def get_A_pix(name):
    path = f'./ControlNet/sampling/{name}/statistics.txt'

    # Initialize variables
    A_mu = None
    max_shape_width = None

    # Open and read the file
    try:
        with open(path, 'r') as file:
            lines = file.readlines()

            # Process each line to find the required values
            for line in lines:
                if 'A_mu:' in line:
                    A_mu = float(line.split(':')[1].strip())
                elif 'Shape:' in line:
                    shape_values = line.split(':')[1].strip().strip('()').split(',')
                    max_shape_width = max(int(shape_values[0].strip()), int(shape_values[1].strip()))

    except FileNotFoundError:
        print(f"File not found: {path}")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

    return A_mu, max_shape_width



def sample_vid(name, cuda_index):
    # Fixed parameters
    prompt = "cell, microscopy, image"
    a_prompt = ''
    n_prompt = ''
    num_samples = 1
    image_resolution = 512
    ddim_steps = 50
    guess_mode = False
    strength = 1.0
    scale = 9.0
    seed = -1  # Set a fixed seed or use a random one
    eta = 0.0
    low_threshold = 100
    high_threshold = 200

    base_dir = f'./ControlNet/sampling/{name}/'

    #if torch.cuda.is_available():
    #    device = torch.device(f"cuda:{cuda_index}")
    #else:
    #    device = torch.device("cpu")
    device = torch.device(f"cuda:{cuda_index}")
    #print('dev',device)
    torch.cuda.set_device(device)

    # Step 1: Create a list of all folders that match the pattern of being a run
    run_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and not f.endswith('_GT')]

    for run in run_folders:
        # Derived and fixed paths
        cond_dir = os.path.join(base_dir, f"{run}_GT/COND/")
        input_dir = os.path.join(base_dir, f"{run}_GT/ID/")
        output_dir = os.path.join(base_dir, run)
        id_path = os.path.join(base_dir, f"{run}_GT/TRA/")
        res_path = os.path.join(base_dir, run)

        # Ensure directories exist and are clean
        if not os.path.exists(cond_dir):
            os.makedirs(cond_dir)
        delete_files_in_folder(output_dir)
        delete_files_in_folder(cond_dir)

        # Step 2: Process initial picture
        num = make_init_pic(name, device,input_dir, output_dir, prompt, a_prompt, n_prompt, num_samples, 
                            image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, 
                            low_threshold, high_threshold)

        # Step 3: Generate video frames
        make_vid(name, device, num, id_path, res_path, cond_dir, prompt, a_prompt, n_prompt, num_samples, 
                 image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)


