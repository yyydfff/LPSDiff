import os
import matplotlib.pyplot as plt
import torchvision
import torch 
import torch.nn as nn
from torch import optim
import logging
import shutil
from torchmetrics.functional import structural_similarity_index_measure
from kornia.filters import get_gaussian_kernel2d, filter2d
import torch.nn.functional as F

from torchvision.utils import save_image



def mkdirs(paths):

    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):

    if not os.path.exists(path):
        os.makedirs(path)
def make_dir(path, refresh=False):
    

    try: os.mkdir(path)
    except: 
        if(refresh): 
            shutil.rmtree(path)
            os.mkdir(path)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger


def save_images(images,root,phase,index,normalize=False):
    # numbers=images.shape[2]
    images=images.permute(1,0,2,3)
    saveroot=root+'/'+str('%02d' % index)+'-'+phase+'.png'
    torchvision.utils.save_image(images,saveroot,padding = 0,normalize=normalize)


def create_optimizer(opt,model):
    learning_rate = opt.learning_rate
    weight_decay =opt.weight_decay
    betas = opt.betas
    # weight_decay = optimizer_config.get('weight_decay', 0)
    # betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay,amsgrad=False)
    return optimizer


def crop_center(img,cropx,cropy,cropz):
    z,y,x = img.shape[2],img.shape[3],img.shape[4]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)  
    startz = z//2-(cropz//2)
    return img[:,:,startz:startz+cropz,starty:starty+cropy,startx:startx+cropx]



def ssim_xy(input,target):
    assert input.size() == target.size()
    b,c,x,y=input.shape
    input=input.reshape(-1,1,x,y)
    target=target.reshape(-1,1,x,y)
    return structural_similarity_index_measure(input,target)





HU_MIN = -1024.0
HU_MAX = 3072.0


USE_METRIC_WINDOW = True
WINDOW_MIN = -1000.0
WINDOW_MAX = 1000.0


def denorm_to_hu(x: torch.Tensor) -> torch.Tensor:

    return x * (HU_MAX - HU_MIN) + HU_MIN


def prepare_metric_hu(input: torch.Tensor, target: torch.Tensor):

    input_hu = denorm_to_hu(input)
    target_hu = denorm_to_hu(target)

    if USE_METRIC_WINDOW:
        input_hu = torch.clamp(input_hu, WINDOW_MIN, WINDOW_MAX)
        target_hu = torch.clamp(target_hu, WINDOW_MIN, WINDOW_MAX)
        data_range = WINDOW_MAX - WINDOW_MIN
    else:
        input_hu = torch.clamp(input_hu, HU_MIN, HU_MAX)
        target_hu = torch.clamp(target_hu, HU_MIN, HU_MAX)
        data_range = HU_MAX - HU_MIN

    return input_hu, target_hu, data_range







def compute_psnr2D(input: torch.Tensor, target: torch.Tensor, max_val: float = None) -> torch.Tensor:
    if not torch.is_tensor(input) or not torch.is_tensor(target):
        raise TypeError(f"Expected 2 torch tensors but got {type(input)} and {type(target)}")

    if input.shape != target.shape:
        raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")

    input_hu, target_hu, data_range = prepare_metric_hu(input, target)

    b, c, h, w = input_hu.shape
    input_hu = input_hu.reshape(-1, 1, h, w)
    target_hu = target_hu.reshape(-1, 1, h, w)

    mse_val = F.mse_loss(input_hu, target_hu, reduction='mean')
    data_range_tensor = torch.tensor(data_range, device=input.device, dtype=input.dtype)

    return 10 * torch.log10((data_range_tensor * data_range_tensor) / mse_val)


def compute_ssim(img1, img2, window_size=11, reduction: str = "mean", max_val: float = None, full: bool = False):

    img1, img2, data_range = prepare_metric_hu(img1, img2)

    window = get_gaussian_kernel2d((window_size, window_size), (1.5, 1.5))
    window = window.unsqueeze(0)
    window = window.requires_grad_(False)

    assert img1.size() == img2.size()

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    b, c, h, w = img1.shape
    img1 = img1.reshape(b * c, 1, h, w)
    img2 = img2.reshape(b * c, 1, h, w)

    tmp_kernel = window.to(device=img1.device, dtype=img1.dtype)

    if tmp_kernel.dim() == 4:
        tmp_kernel = tmp_kernel.squeeze(0)

    def _filter2d(x):
        return filter2d(x, tmp_kernel)

    mu1 = _filter2d(img1)
    mu2 = _filter2d(img2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _filter2d(img1 * img1) - mu1_sq
    sigma2_sq = _filter2d(img2 * img2) - mu2_sq
    sigma12 = _filter2d(img1 * img2) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_map.view(b, c, h, w)

    if reduction != 'none':
        ssim_map = torch.clamp(ssim_map, min=0, max=1)
        if reduction == "mean":
            ssim_map = torch.mean(ssim_map)
        elif reduction == "sum":
            ssim_map = torch.sum(ssim_map)

    if full:
        cs = torch.mean((2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        return ssim_map, cs

    return ssim_map



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        b, c, h, w = y.size()
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss/(c*b*h*w*t)

def compute_rmse(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    RMSE in HU.
    """
    input_hu, target_hu, _ = prepare_metric_hu(input, target)
    return torch.sqrt(F.mse_loss(input_hu, target_hu))