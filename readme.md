# SynCellFactory

## Introduction
In this directory the codebase for SynCellFactory with an executable example is presented.

## Prerequisites
- A single A100 40GB GPU is required for sampling and training as described in the paper.
- Python 3.x

## Installation
The dependencies can be installed with 

pip install -r requirements.txt



## Usage
### Running the Provided Example
1. Download `DIC-C2DH-HeLa` and `DIC-C2DH-HeLa_track` folders from https://drive.google.com/drive/folders/1zvqZiFwMUMwReB1jdSMTUtTKqiX2KGRG?usp=drive_link
2. Place both folders into `./ControlNet/models/`.
3. Start the SynCellFactory sampling process:
   ```bash
   python main.py DIC-C2DH-HeLa
   
### Configurations
This will load the predefined sampling parameters from ./configs/DIC-C2DH-HeLa.json which are:


    "name": "DIC-C2DH-HeLa", #name of the dataset, should be chosen to be the original name from the CTC
    "num_vid": 3,	      # How many videos should be generated
    "num_timeframes": 12,    # How many timeframes should be generated per video
    "sp_prob": 0.15,         # Parameter to scale the frequency of splitting events
    "d_mitosis": 4,	      # How long is the visual mitosis cycle
    "n_cells": [5,7],	      # How many cells should be in the last timeframe; this will be uniformly sampled from the given range and should be in the same order as for the training set
    "train_CNet": false,     # Should new ControlNets be trained on this dataset; setting this to true will start the automated training process
    "cuda_index": 1,         # GPU index for sampling and training
    "cellpose_path": "",     # Path to the finetuned Cellpose model; in this example we use the basemodel CPx for simplicity
    "cellpose_modelname": "CPx" #name of the finetuned Cellpose model; in this example we use the basemodel CPx for simplicity
    
Parameters can be changed. 
New config files can be created for all datasets in the same manner. 


## Output
The generated videos in CellTrackingChallenge format can be found in './Outputs'

## Other Datasets
To apply SynCellFactory to other datasets, download the official training datasets (consisting of two training videos; one used for training and one for testing) from http://celltrackingchallenge.net/2d-datasets/ and place them in ./Inputs in the corresponding 'train' and 'test' folders.


## Training
Automated training can be started by downloading control _ sd15 _ cell.ckpt from google drive and place it into `./ControlNet/models/`. This ckpt serves as the initial ckpt for the automated training. Create a corresponding .json file in the './configs' subfolder for the dataset of interest and set "train_CNet" to true. 




