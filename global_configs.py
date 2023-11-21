import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROGRAM"] = "train.py"

DEVICE = torch.device("cuda:0")

ACOUSTIC_DIM = 64
VISUAL_DIM = 512
TEXT_DIM = 768

XLNET_INJECTION_INDEX = 1
