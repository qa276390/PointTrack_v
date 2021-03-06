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

    save=True,
    #save_dir='./outputs/tracking/tracks_car_pointtrack_val_transformer_weighed_loss/',
    save_dir='./outputs/tracking/tracks_car_pointtrack_val_residual_nearby_5',
    #checkpoint_path='./weights/car_finetune_tracking/checkpoint.pth',
    checkpoint_path='./weights/car_finetune_tracking_transformer_residual_nearby_5/best_iou_model.pth85.49_0.002',
    #checkpoint_path='./weights/car_finetune_tracking_transformer_residual/best_iou_model.pth85.47_0.002',
    #checkpoint_path='./weights/car_finetune_tracking_transformer_with_triplet/best_iou_model.pth85.26_0.0002',
    #checkpoint_path='./weights/car_finetune_tracking_transformer_freeze/best_iou_model.pth85.33_0.0002',
    #run_eval=False,
    run_eval=True,

    dataset= {
        'name': 'mots_track_val_env_offset',
        'kwargs': {
            'root_dir': kittiRoot,
            'mode': 'val',
            'num_points': 1500,
            'box': True,
            'gt': False,
            'category': True,
            'ex':0.2
        },
        'batch_size': 1,
        'workers': 32
    },

    model={
        #'name': 'tracker_offset_emb',
        'name': 'tracker_offset_emb_transformer',
        'kwargs': {
            'num_points': 1000,
            'margin': 0.2,
            'border_ic': 3,
            'env_points': 500,
            'outputD': 32,
            'category': True,
            'residual': True # add by vtsai01
        }
    },
    max_disparity=192.0,
    with_uv=True
)


def get_args():
    return copy.deepcopy(args)
