from share import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from pytorch_lightning.callbacks.base import Callback
import subprocess
from config_loader import load_config
import os

conf_p = os.environ.get('CONFIG_PATH')
conf = load_config(conf_p)
resume_path = conf["resume_path"]
ckpt_save_path = conf["ckpt_save_path"]
gpu_train = conf["gpu_train"]
gpu_samp = conf["gpu_samp"]
name = conf["name"]
m_steps = conf["max_steps"]

prompt = 'cell, microscopy, image'

class ExternalScriptCallback(Callback):
    def on_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        subprocess.call(["python", "val_sampling.py", "--ckpt_path", ckpt_save_path + '/last.ckpt', "--prompt", prompt, "--epoch", str(epoch), "--gpu", str(gpu_samp)])

# Define a ModelCheckpoint callback.
checkpoint_callback = ModelCheckpoint(
    dirpath=ckpt_save_path,
    save_weights_only=True,
    verbose=True,
    filename='last-{epoch:02d}',
    save_last=True
)

logger_freq = 300
logger = ImageLogger(batch_frequency=logger_freq)
callbacks = [logger, checkpoint_callback]

def main():
    learning_rate = 5e-6
    sd_locked = False
    only_mid_control = False

    model = create_model('./ControlNet/models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    trainer = pl.Trainer(
        gpus=[gpu_train],
        precision=16,
        callbacks=callbacks,
        max_steps=m_steps
    )

    trainer.fit(model)

if __name__ == '__main__':
    main()
