
import os
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Setup script for training.")
    parser.add_argument("conf_path", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    # Set the environment variable with the passed config path
    os.environ['CONFIG_PATH'] = args.conf_path
    
    script_path = os.path.join(os.path.dirname(__file__), "train.py")
    subprocess.call(["python", script_path])
    
    print("Config Path:", os.environ.get('CONFIG_PATH'))

if __name__ == "__main__":
    main()

