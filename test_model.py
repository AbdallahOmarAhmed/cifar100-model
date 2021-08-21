import torch
import torchvision
import torchvision.transforms as trans
import torch.nn as nn
import numpy as np
from cifar100 import cifar100Net
# change the path to be like your model path
path = "your_model_path/model.pth"

# load the model
model = cifar100Net(*args, **kwargs)
model.load_state_dict(torch.load(path))
model.eval()

# Good luck :)
