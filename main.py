import json
import sys
import os
from motion_module.motion_main import motion_run
from create_train_data.train_main import train_run
from ControlNet.create_vid import sample_vid




def load_configuration(config_name):
    config_path = os.path.join('configs', f'{config_name}.json')
    try:
        with open(config_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Invalid JSON in the configuration file: {config_path}")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <config_name>")
        sys.exit(1)

    config_name = sys.argv[1]
    config = load_configuration(config_name)

    # Extract parameters
    name = config.get('name')
    num_vid = config.get('num_vid')
    num_timeframes = config.get('num_timeframes')
    sp_prob = config.get('sp_prob')
    d_mitosis = config.get('d_mitosis')
    n_cells = config.get('n_cells')
    train_CNet = config.get('train_CNet')
    cuda_index = config.get('cuda_index')

    input_path= './Inputs/'+name+'/train/'
    mm_path = './ControlNet/sampling/'+name
    path_stats = mm_path+'/statistics.txt'


    save_path_train='./ControlNet/training/'
    save_dir_train=save_path_train+name+'_train'


    motion_run(input_path,mm_path,num_vid,sp_prob,num_timeframes,n_cells,d_mitosis)
    if train_CNet:
        train_run(path_stats,input_path,name,save_dir_train,d_mitosis)
    sample_vid(name,cuda_index)

if __name__ == "__main__":
    main()