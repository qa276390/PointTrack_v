"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os

from PIL import Image

import torch
from utils import transforms as my_transforms
from config import *

args = dict(

    cuda=True,
    display=False,
    display_it=5,

    save=True,
    save_dir='./weights/car_finetune_tracking_transformer_residual_nearby_5',
    eval_config='car_test_tracking_val',
    resume_path='./weights/car_finetune_tracking/checkpoint.pth',
    #resume_path='./weights/car_finetune_tracking_transformer_with_triplet/checkpoint.pth',
    #resume_path='./weights/car_finetune_tracking_transformer_freeze/best_iou_model.pth85.33_0.0002',

    train_dataset = {
        'name': 'mots_track_cars_train_transformer',
        'kwargs': {
            'root_dir': kittiRoot,
            'mode': 'train',
            'size': 500,
            'num_points': 1500,
            'shift': True,
            'sample_num': 24,
            'nearby': 5, # vtsai01
            'category': True
        },
        'batch_size': 1,
        #'workers': 1
        'workers':32
    },

    model = {
        'name': 'tracker_offset_emb_transformer',
        'kwargs': {
            'num_points': 1000,
            'margin': 0.2,
            'border_ic': 3,
            'env_points': 500,
            'outputD': 32,
            'category': True, 
            'freeze': True, # freeze feature extractor
            'residual': True
        }
    },

    lr=2e-3,
    milestones=[75, 120, 250],
    n_epochs=300,
    start_epoch=1,
    epoch_to_SGD = 150,

    max_disparity=192.0,
    val_interval=5,

    # loss options
    loss_opts={
        'to_center': True,
        'n_sigma': 1,
        'foreground_weight': 10,
    },
    loss_w={
        'w_inst': 1,
        'w_var': 10,
        'w_seed': 1,
    },
)


def get_args():
    return copy.deepcopy(args)
