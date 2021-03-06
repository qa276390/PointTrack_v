"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os, sys
import shutil
import time
from config import *

os.chdir(rootDir)

from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
from config_mots import *
from datasets import get_dataset
from models import get_model
from utils.utils import AverageMeter, Cluster, Logger, Visualizer
from file_utils import remove_key_word
import subprocess
import numpy as np

from torch.utils.tensorboard import SummaryWriter



torch.backends.cudnn.benchmark = True
config_name = sys.argv[1]

args = eval(config_name).get_args()

# Writer will output to ./runs/ directory by default
writer = SummaryWriter(comment = args['save_dir'])

if 'cudnn' in args.keys():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

# train dataloader
train_dataset = get_dataset(args['train_dataset']['name'], args['train_dataset']['kwargs'])
print(args['train_dataset']['kwargs'])
train_dataset_it = torch.utils.data.DataLoader(
    train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# set model
model = get_model(args['model']['name'], args['model']['kwargs'])
model.init_output(args['loss_opts']['n_sigma'])

# set device
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)
free_gpu_id = get_freer_gpu()
print('free gpu id', free_gpu_id)
device = torch.device(f"cuda:{free_gpu_id}" if args['cuda'] else "cpu")


model = torch.nn.DataParallel(model, device_ids=[int(free_gpu_id)])
model.to(f'cuda:{model.device_ids[0]}')
#model = torch.nn.DataParallel(model).to(device)

# set optimizer
optimizer_1 = torch.optim.Adam(
    model.parameters(), lr=args['lr'], weight_decay=1e-4)
optimizer_2 = torch.optim.SGD(
    model.parameters(), lr=args['lr'], weight_decay=1e-4)
optimizer = optimizer_1

def lambda_(epoch):
    return pow((1 - ((epoch) / args['n_epochs'])), 0.9)


# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_,)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['milestones'], gamma=0.1)

# clustering
cluster = Cluster()

# Visualizer
visualizer = Visualizer(('image', 'pred', 'sigma', 'seed'))

# Logger
logger = Logger(('train', 'val', 'iou'), 'loss')

# resume
start_epoch = 1
best_iou = 0
best_seed = 10
if 'resume_path' in args.keys() and args['resume_path'] is not None and os.path.exists(args['resume_path']):
    print('Resuming model from {}'.format(args['resume_path']))
    state = torch.load(args['resume_path'])
    if 'start_epoch' in args.keys():
        start_epoch = args['start_epoch']
    elif 'epoch' in state.keys():
        start_epoch = state['epoch'] - 1
    else:
        start_epoch = 1
    if 'optimizer_state_dict' in args.keys():
        optimizer.load_state_dict(state['optimizer_state_dict'])
    if 'scheduler' in args.keys():
        scheduler.load_state_dict(state['scheduler'])
    # best_iou = state['best_iou']
    for kk in state.keys():
        if 'state_dict' in kk:
            state_dict_key = kk
            break
    new_state_dict = state[state_dict_key]
    print('state.keys:', state.keys())
    if not 'state_dict_keywords' in args.keys():
        try:
            model.load_state_dict(new_state_dict, strict=True)
        except:
            print('resume checkpoint with strict False')
            model.load_state_dict(new_state_dict, strict=False)
    else:
        new_state_dict = remove_key_word(state[state_dict_key], args['state_dict_keywords'])
        model.load_state_dict(new_state_dict, strict=False)
    try:
        logger.data = state['logger_data']
    except:
        pass


def train(epoch):
    # define meters
    loss_meter = AverageMeter()
    loss_emb_meter = AverageMeter()

    # put model into training mode
    model.train()

    if epoch > args['epoch_to_SGD']:
        optimizer = optimizer_2
    else: 
        optimizer = optimizer_1

    for param_group in optimizer.param_groups:
        print('learning rate: {}'.format(param_group['lr']))

    for i, sample in enumerate(tqdm(train_dataset_it)):
        points = sample['points']
        xyxys = sample['xyxys']
        labels = sample['labels']
        fstamps = sample['framestamp']
        emb_loss, transformer_loss = model(points, labels, xyxys, fstamps)

        loss = emb_loss + transformer_loss
        if loss.item() > 0:
            loss_emb_meter.update(emb_loss.mean().item())
            loss_meter.update(transformer_loss.mean().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_meter.avg, loss_emb_meter.avg


def val(epoch):
    state = {
        'epoch': epoch,
        'best_iou': best_iou,
        'best_seed': best_seed,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'logger_data': logger.data
    }
    file_name = os.path.join(args['save_dir'], 'checkpoint.pth')
    torch.save(state, file_name)

    val_name = args['eval_config']
    runfile = "test_tracking.py"
    p = subprocess.run([pythonPath, "-u", runfile,
                        val_name], stdout=subprocess.PIPE, cwd=rootDir)

    val_args = eval(val_name).get_args()
    save_val_dir = val_args['save_dir'].split('/')[1:]
    p = subprocess.run([pythonPath, "-u", "eval.py",
                        os.path.join(rootDir, *save_val_dir), kittiRoot + "instances", "val.seqmap"],
                       stdout=subprocess.PIPE, cwd=os.path.join(rootDir, "datasets/mots_tools/mots_eval"))
    pout = p.stdout.decode("utf-8")
    if 'person' in args['save_dir']:
        class_str = "Evaluate class: Pedestrians"
    else:
        class_str = "Evaluate class: Cars"
    pout = pout[pout.find(class_str):]
    print(pout[pout.find('all   '):][6:126].strip())
    acc = pout[pout.find('all   '):][6:26].strip().split(' ')[0]

    return 0.0, float(acc)


def save_checkpoint(state, is_best, iou_str, is_lowest=False, name='checkpoint.pth'):
    print('=> saving checkpoint')
    if 'save_name' in args.keys():
        file_name = os.path.join(args['save_dir'], args['save_name'])
    else:
        file_name = os.path.join(args['save_dir'], f"{name}.pth")
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_iou_model.pth' + iou_str))
    if is_lowest:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_seed_model.pth'))

#val_loss, val_iou = val(start_epoch-1)
#print('===> val loss: {:.4f}, val iou: {:.4f}'.format(val_loss, val_iou))
best_train_loss = 1e8

for epoch in range(start_epoch, args['n_epochs']):

    print('Starting epoch {}'.format(epoch))
    # scheduler.step(epoch)
    # if epoch == start_epoch:
    #     print('Initial eval')
    #     val_loss, val_iou = val(epoch)
    #     print('===> val loss: {:.4f}, val iou: {:.4f}'.format(val_loss, val_iou))

    train_loss, emb_loss = train(epoch)
    print('===> train loss: {:.4f}, train emb loss: {:.4f}'.format(train_loss, emb_loss))
    logger.add('train', train_loss)
    writer.add_scalar('Loss/train', train_loss, epoch)

    

    if 'val_interval' not in args.keys() or epoch % args['val_interval'] == 0 or epoch == 1:
        val_loss, val_iou = val(epoch)
        print('===> val loss: {:.4f}, val iou: {:.4f}'.format(val_loss, val_iou))
        logger.add('val', val_loss)
        writer.add_scalar('Loss/val', val_loss, epoch)
        logger.add('iou', val_iou)
        writer.add_scalar('IoU/val', val_iou, epoch)
        # logger.plot(save=args['save'], save_dir=args['save_dir'])

        is_best = val_iou >= best_iou
        best_iou = max(val_iou, best_iou)

        if args['save']:
            state = {
                'epoch': epoch,
                'best_iou': best_iou,
                'best_seed': best_seed,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'logger_data': logger.data,
                'scheduler': scheduler
            }
            for param_group in optimizer.param_groups:
                lrC = str(param_group['lr'])

            save_checkpoint(state, is_best, str(best_iou) + '_' + lrC, is_lowest=False)
    scheduler.step(epoch)