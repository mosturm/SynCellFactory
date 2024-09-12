# SynCellFactory

## Introduction
This directory contains the SynCellFactory codebase along with an executable example.

## Prerequisites
- A single A100 40GB GPU is required for sampling and training as described in the paper.
- Python 3.x
- Linux (Tested on Ubuntu 18.04)

## Installation
1. Create a new Python environment:

   ```bash
   python -m venv env
   ```

2. Activate the environment:

   ```bash
   source env/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```




## Usage
### Running the Provided Example
1. Download the `DIC-C2DH-HeLa` and `DIC-C2DH-HeLa_track` folders (which contain the trained CN-Pos and CN-Mov checkpoints for DIC-C2DH-HeLa) from https://drive.google.com/drive/folders/1zvqZiFwMUMwReB1jdSMTUtTKqiX2KGRG?usp=drive_link
2. Place both folders into `./ControlNet/models/` in the following structure: 
```
SynCellFactory/
└── ControlNet/
    └── models/
        ├── DIC-C2DH-HeLa/
        │   └── last.ckpt
        └── DIC-C2DH-HeLa_track/
            └── last.ckpt
```
3. Download the official 2D datasets from the CTC : http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip 
4. Place the train and test datasets from the CTC (labeled `01` and `02` ) in `./Inputs/DIC-C2DH-HeLa/` in the following structure: 
```
SynCellFactory/
└── Inputs/
    └── DIC-C2DH-HeLa/
        ├── test/
        │   ├── 01/
        │   ├── 01_GT/
        │   └── 01_ST/
        └── train/
            ├── 02/
            ├── 02_GT/
            └── 02_ST/
```
5. Start the SynCellFactory sampling process:
   ```bash
   python main.py DIC-C2DH-HeLa
   
### Configurations
This will load the predefined sampling parameters from `./configs/DIC-C2DH-HeLa.json` as follows::


    "name": "DIC-C2DH-HeLa", #name of the dataset, should be chosen to be the original name from the CTC
    "num_vid": 3,	      # The number of videos to be generated
    "num_timeframes": 12,    # How many timeframes should be generated per video
    "sp_prob": 0.15,         # Parameter to scale the frequency of splitting events
    "d_mitosis": 4,	      # How long is the visual mitosis cycle
    "n_cells": [5,7],	      # How many cells should be in the last timeframe; this will be uniformly sampled from the given range and should be in the same order as for the training set
    "train_CNet": false,     # Should new ControlNets be trained on this dataset; setting this to true will start the automated training process
    "cuda_index": 0,         # GPU index for sampling and training
    "cellpose_path": "",     # Path to the finetuned Cellpose model; in this example we use the basemodel CPx for simplicity
    "cellpose_modelname": "CPx" #name of the finetuned Cellpose model; in this example we use the basemodel CPx for simplicity
    
Parameters can be changed. 
New config files can be created for all datasets in the same manner by copying and renaming the .json file in the `./configs/` folder. 


## Output
The generated videos in CellTrackingChallenge format can be found in `./Outputs`

## Other Datasets
To apply SynCellFactory to other datasets, create a corresponding config file in ./configs named after the dataset of interest. Then, follow the same procedure as described above for DIC-C2DH-HeLa, but rename everything accordingly. For now, the following checkpoints are available for demonstration purposes on https://drive.google.com/drive/folders/1zvqZiFwMUMwReB1jdSMTUtTKqiX2KGRG?usp=drive_link:

    Fluo-C2DL-Huh7
    Fluo-N2DH-GOWT1
    DIC-C2DH-HeLa

## Training
Automated training can be started by downloading control _ sd15 _ cell.ckpt from Google Drive (https://drive.google.com/drive/folders/1zvqZiFwMUMwReB1jdSMTUtTKqiX2KGRG?usp=drive_link) and place it into `./ControlNet/models/` in the following structure: 

```
SynCellFactory/
└── ControlNet/
    └── models/
        └── control_sd15_cell.ckpt
```


This ckpt serves as the initial ckpt for the automated training. Create a corresponding .json file in the './configs' subfolder named after the dataset of interest and set "train_CNet" to true. 
To start training, run the following command:
   ```bash
   python main.py Dataset_name
   ```

This will automatically start the training of the dataset and will sample new videos once the training is finished.

Ensure that your datasets follow the Cell Tracking Challenge format and adhere to the folder structure explained in Step 4.




