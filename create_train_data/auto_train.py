import json
import subprocess
import numpy as np
from tqdm import tqdm

def generate_config_and_path(name, resume_path, ckpt_save_path, gpu_train, gpu_samp, max_steps):
    # Create the configuration dictionary
    config_params = {
        "name": name,
        "resume_path": resume_path,
        "ckpt_save_path": ckpt_save_path,
        "gpu_train": gpu_train,
        "gpu_samp": gpu_samp,
        "max_steps": max_steps
    }

    # Generate the file path based on the name
    config_file_path = f'./ControlNet/configs/{name}.json'

    return config_params, config_file_path

def write_config_to_file(config_params, config_file_path):
    with open(config_file_path, 'w') as file:
        json.dump(config_params, file, indent=4)

def run_training(config_file):
    command = f"python ./ControlNet/setup.py '{config_file}'"
    subprocess.run(command, shell=True)

def create_resume_path_variations(base_string):
    # Constructing different variations of the resume_path
    resume_path_variations = {
        "entry1": './ControlNet/models/control_sd15_cell.ckpt',
        "entry2": f'./ControlNet/models/{base_string}_BM/last.ckpt',
        "entry3": f'./ControlNet/models/{base_string}_BM/last.ckpt',
        "entry4": f'./ControlNet/models/{base_string}_BM_track/last.ckpt'
    }
    return resume_path_variations

def create_name_variations(base_string, max_shape_width, min_shape_height):
    # Constructing different variations of the name
    variations = {
        "entry1": f"{base_string}_{int(max_shape_width/2)}_{int(min_shape_height/2)}",
        "entry2": f"{base_string}_track_{int(max_shape_width/2)}_{int(min_shape_height/2)}",
        "entry3": f"{base_string}_{int(max_shape_width)}_{int(min_shape_height)}",
        "entry4": f"{base_string}_track_{int(max_shape_width)}_{int(min_shape_height)}"
    }
    return variations

def create_ckpt_save_path_variations(base_string):
    # Constructing different variations of the ckpt_save_path
    ckpt_save_path_variations = {
        "entry1": f'./ControlNet/models/{base_string}_BM',
        "entry2": f'./ControlNet/models/{base_string}_BM_track',
        "entry3": f'./ControlNet/models/{base_string}',
        "entry4": f'./ControlNet/models/{base_string}_track'
    }
    return ckpt_save_path_variations

def create_max_steps_variations(n_cells):
    if np.max(n_cells) < 100:
        max_steps_variations = {
            "entry1": 30000,
            "entry2": 15000,
            "entry3": 3000,
            "entry4": 3000
        }
        
    elif 100 <= np.max(n_cells) < 1000:
        max_steps_variations = {
            "entry1": 60000,
            "entry2": 30000,
            "entry3": 7000,
            "entry4": 7000
        }
    else:
        print("SynCellFactory not suitable for this amount of cells")
        raise Exception("Aborting due to unsuitable cell count")

    return max_steps_variations

def create_configuration_list(name,n_cells,cuda_index):
    stat_path=f'./ControlNet/sampling/{name}/statistics.txt'
    try:
        with open(stat_path, 'r') as file:
            lines = file.readlines()

            # Process each line to find the required values
            for line in lines:
                if 'Shape:' in line:
                    shape_values = line.split(':')[1].strip().strip('()').split(',')
                    max_shape_width = max(int(shape_values[0].strip()), int(shape_values[1].strip()))
                    min_shape_heigth = min(int(shape_values[0].strip()), int(shape_values[1].strip()))

    except FileNotFoundError:
        print(f"File not found: {stat_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
    
    constructed_names = create_name_variations(name, max_shape_width, min_shape_heigth)
    res_paths= create_resume_path_variations(name)
    ckpt_paths=create_ckpt_save_path_variations(name)
    max_steps_l = create_max_steps_variations(n_cells)


    configurations = []
    for i in range(1, 5):
        config = {
            "name": constructed_names[f"entry{i}"],
            "resume_path": res_paths[f"entry{i}"],
            "ckpt_save_path": ckpt_paths[f"entry{i}"],
            "gpu_train": cuda_index,  # Set to cuda index
            "gpu_samp": 0,            # Always set to 0
            "max_steps": max_steps_l[f"entry{i}"]
            }
        configurations.append(config)

    return configurations




def auto_train_main(name,n_cells,cuda_index):
    configurations=create_configuration_list(name,n_cells,cuda_index)
    #print('config_list_auto_train',configurations)
    #print(stop)
    count=0
    for config in tqdm(configurations, desc="Automated Training of 4 ControlNets started, this could take a while..."):
        config_params, config_file_path = generate_config_and_path(**config)
        write_config_to_file(config_params, config_file_path)
        run_training(config_file_path)
        count=count+1






