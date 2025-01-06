import resnet_design2 as models # Design 2
import argparse
import os
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
from torchsummary import summary

model_in = models.resnext50_32x4d()
summary(model_in.cuda(), (2,3, 224, 224))

#model_in = models.resnet2_18()
#summary(model_in.cuda(), (2,3, 224, 224))
