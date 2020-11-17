"""
Author: Zhenbo Xu
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os, sys
import time
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm
from config_mots import *
from datasets import get_dataset
from models import get_model
from utils.mots_util import *
from config import *
import subprocess
from torch.utils.tensorboard import SummaryWriter # visualization vtsai01
import datetime;

ts = datetime.datetime.now().strftime("%m%d-%H%M")


# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 0
torch.manual_seed(seed)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

config_name = sys.argv[1]
args = eval(config_name).get_args()
max_disparity = args['max_disparity']


# default `log_dir` is "runs" - we'll be more specific here
dataset_ = args['dataset']['name']
writer = SummaryWriter(f'runs/{config_name}-{ts}') # visualization vtsai01

if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

# set device
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)
free_gpu_id = get_freer_gpu()
print('free gpu id', free_gpu_id)
#device = torch.device(f"cuda:{free_gpu_id}" if args['cuda'] else "cpu")
#torch.cuda.set_device(device)

# dataloader
dataset = get_dataset(
    args['dataset']['name'], args['dataset']['kwargs'])
print('dataset',  args['dataset']['name'],  args['dataset']['kwargs'])
dataset_it = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True if args['cuda'] else False)

# load model
model = get_model(args['model']['name'], args['model']['kwargs'])

# Parallel
model = torch.nn.DataParallel(model, device_ids=[int(free_gpu_id)])
print('d_ids', model.device_ids[0])
model.to(f'cuda:{model.device_ids[0]}')

# load snapshot
if os.path.exists(args['checkpoint_path']):
    state = torch.load(args['checkpoint_path'])
    model.load_state_dict(state['model_state_dict'], strict=False)
    print('Load dict from %s' % args['checkpoint_path'])
else:
    #assert(False, 'checkpoint_path {} does not exist!'.format(args['checkpoint_path']))
    print(args['checkpoint_path'])



model.eval()


def prepare_img(image):
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, torch.Tensor):
        image.squeeze_()
        image = image.numpy()

    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] in {1, 3}:
            image = image.transpose(1, 2, 0)
        return image


dColors = [(128, 0, 0), (170, 110, 40), (128, 128, 0), (0, 128, 128), (0, 0, 128), (230, 25, 75), (245, 130, 48)
        , (255, 225, 25), (210, 245, 60), (60, 180, 75), (70, 240, 240), (0, 130, 200), (145, 30, 180), (240, 50, 230)
        , (128, 128, 128), (250, 190, 190), (255, 215, 180), (255, 250, 200), (170, 255, 195), (230, 190, 255), (255, 255, 255)]
print('args[\'save_dir\']', args['save_dir'])

use_transformer = True if 'transformer' in args['model']['name'] else False
export_emb =  False # visualization vtsai01
mask_iou = False

if use_transformer:
    trackHelper = TrackHelperTransformer(model, args['save_dir'], model.module.margin, alive_car=30, car=args['car'] if 'car' in args.keys() else True,
                          mask_iou=mask_iou, use_ttl=False, ttl=2, export_emb=export_emb, tb_writer=writer)
else:
    trackHelper = TrackHelper(args['save_dir'], model.module.margin, alive_car=30, car=args['car'] if 'car' in args.keys() else True,
                          mask_iou=mask_iou, use_ttl=False, ttl=2, export_emb=export_emb, tb_writer=writer) # use_ttl is optional @vtsai01
print("+"*10)



with torch.no_grad():
    print(len(dataset_it))

    for sample in tqdm(dataset_it):
        subf, frameCount = sample['name'][0][:-4].split('/')[-2:] # subf: folder; frameCount: NO. of frame
        frameCount = int(float(frameCount))
        filepath = sample['name'][0]
        # MOTS forward with tracking
        points = sample['points']
        if len(points) < 1:
            embeds = np.array([])
            masks = np.array([])
            xyxys_np = np.array([])
        else:
            masks = sample['masks'][0]
            xyxys = sample['xyxys']
            embeds_t = model(points, None, xyxys, infer=True)
            embeds = embeds_t.cpu().numpy()
            masks = masks.numpy()
            xyxys_np = sample['xyxys'][0].numpy()
      
        # do tracking
        trackHelper.tracking(subf, frameCount, embeds, masks, xyxys_np, filepath)
    
    if export_emb:
        trackHelper.export_embeddings()
    writer.close()
    trackHelper.export_last_video()

if 'run_eval' in args.keys() and args['run_eval']:
    # run eval
    save_val_dir = args['save_dir'].split('/')[1:]
    p = subprocess.run([pythonPath, "-u", "eval.py",
                        os.path.join(rootDir, *save_val_dir), kittiRoot + "instances", "val.seqmap"],
                                   stdout=subprocess.PIPE, cwd=rootDir + "/datasets/mots_tools/mots_eval")
    print(p.stdout.decode("utf-8"))
