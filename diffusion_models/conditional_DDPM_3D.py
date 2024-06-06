'''this script is the 3D conditional DDPM'''
import math
import copy
import os
import pandas as pd
import numpy as np
import nibabel as nb
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from skimage.measure import block_reduce

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from Diffusion_for_CT_motion.diffusion_models.attend import Attend
from Diffusion_for_CT_motion.diffusion_models.version import __version__
import Diffusion_for_CT_motion.utils.functions_collection as ff
import Diffusion_for_CT_motion.utils.Data_processing as Data_processing

##### constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

##### helpers functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val): 
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def Upsample3D(dim, dim_out = None, upsample_factor = (2,2,1)):
    return nn.Sequential(
        nn.Upsample(scale_factor = upsample_factor, mode = 'nearest'),
        nn.Conv3d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample3D(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) d -> b (c p1 p2) h w d', p1 = 2, p2 = 2),
        nn.Conv3d(dim * 4, default(dim_out, dim), 1)
    )


class RMSNorm3D(nn.Module):
    '''RMSNorm applies channel-wise normalization to the input tensor, 
    scales the normalized values using the learnable parameter g, 
    and then further scales the result by the square root of the number of input channels. '''
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1 , 1)) # learnable

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

##### sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module): # output dimension is dim
    '''https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/'''
    def __init__(self, dim, theta = 10000): # theta is the n on the guidance webpage, dim is d (Dimension of the output embedding space).
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)  # weights are not 1/(n**(2i/d)), instead it's learnable

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

##### building block modules
class Block3D(nn.Module):  # input dimension is dim, output dimension is dim_out
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)  
        self.act = nn.SiLU()  

    def forward(self, x, scale_shift = None):
        x = self.conv(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock3D(nn.Module): # input dimension is dim , output dimension is dim_out. for time_emb, the input dimension is time_emb_dim, output dimension is dim_out * 2
    '''experience two basic convolution+group_normlization+SiLu blocks, and then add the input to the output of the second block.'''
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2) # fully-connected layer
        ) if exists(time_emb_dim) else None

        self.block1 = Block3D(dim, dim_out, groups = groups)
        self.block2 = Block3D(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)  # scale shift is how we incorporate the time embedding into x

        h = self.block2(h)

        return h + self.res_conv(x)
    
class LinearAttention3D(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm3D(dim)
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv3d(hidden_dim, dim, 1),
            RMSNorm3D(dim)
        )

    def forward(self, x):
        b, c, h, w, d = x.shape  # Added dimension 'd' for depth

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)  # split input into q k v evenly
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h c (x y z)', h=self.heads), qkv)  # h = head, c = dim_head

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)  # matrix multiplication

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y z) -> b (h c) x y z', h = self.heads, x = h, y = w, z = d)
        return self.to_out(out)


class Attention3D(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm3D(dim)
        self.attend = Attend(flash = flash)

        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w, d = x.shape  # Added dimension 'd' for depth

        x = self.norm(x) 

        qkv = self.to_qkv(x).chunk(3, dim=1)  # split input into q k v evenly
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z-> b h (x y z) c', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x = h, y = w, z = d)
        return self.to_out(out)

##### Deep learning model
class Unet3D(nn.Module):
    def __init__(
        self,
        init_dim,
        channels,
        conditional_diffusion,

        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        self_condition = False,   # use the prediction from the previous iteration as the condition of next iteration
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = (None, None, None, True),
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions
        self.self_condition = self_condition
        self.conditional_diffusion = conditional_diffusion
        self.channels = channels
        input_channels = channels + (1 if self.conditional_diffusion else 0)  # add one channel for the condition

        self.init_conv = nn.Conv3d(input_channels, init_dim, 5, padding = 2) # if want input and output to have same dimension, Kernel size to any odd number (e.g., 3, 5, 7, etc.). Padding to (kernel size - 1) / 2.

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]  # if initi_dim = 16, then [16,32,64,128]
        in_out = list(zip(dims[:-1], dims[1:])) 
        # [(16,16), (16, 32), (32, 64), (64, 128)]. Each tuple in in_out represents a pair of input and output dimensions for different stages in a neural network

        block_klass = partial(ResnetBlock3D, groups = resnet_block_groups)  # Here, block_klass is being defined as a new function that is essentially a ResnetBlock, but with the groups argument set to resnet_block_groups. This means that when you call block_klass, you only need to provide the remaining arguments that ResnetBlock expects (such as dim and dim_out), and groups will be automatically set to the value of resnet_block_groups.

        # time embeddings

        time_dim = init_dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(init_dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = init_dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),  # Gaussian error activation function
            nn.Linear(time_dim, time_dim)
        )

        # attention
        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention3D, flash = flash_attn)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out) # 4

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            if layer_full_attn == True:
                attn_klass = FullAttention
            elif layer_full_attn == False:
                attn_klass = LinearAttention3D

            # in each downsample stage, 
            # we have 4 layers: 2 resnet blocks (doesn't increase the feature number), 1 attention layer, and 1 downsampling layer (downsample x and y by 2, then increase the feature number by 2)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads) if layer_full_attn is not None else nn.Identity(),
                Downsample3D(dim_in, dim_out) if not is_last else nn.Conv3d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            if layer_full_attn == True:
                attn_klass = FullAttention 
            elif layer_full_attn == False:
                attn_klass = LinearAttention3D
          
            # in each upsample stage,
            # we have 4 layers: 2 resnet blocks (does change the feature number), 1 attention layer, and 1 upsampling layer (upsample x and y by 2, then decrease the feature number by 2)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads) if layer_full_attn is not None else nn.Identity(),
                Upsample3D(dim_out, dim_in) if not is_last else  nn.Conv3d(dim_out, dim_in, 5, padding = 2)  
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)  # channels = input channels, learned_variance = False
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(init_dim * 2, init_dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv3d(init_dim, self.out_dim, 1)  # output channel is initial channel number

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1) # = 8

    def forward(self, x, time, condition = None, x_self_cond = None):

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        
        # concatenate the condition to the input along the dimension = 1
        if self.conditional_diffusion:
            if exists(condition) == 0:
                raise ValueError('condition is required for conditional diffusion')
            x = torch.cat((x, condition), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)

            x = attn(x) + x
            h.append(x)

            x = downsample(x)
          

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)
        
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)   # h.pop() is the output of the corresponding downsample stage
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)

            x = block2(x, t)

            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        final_image = self.final_conv(x)
      
        return final_image

##### Diffusion Functions
def extract(a, t, x_shape):
    # extract at from a list of a, then add empty axes to match the image dimension
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) 

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion3D(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size_3D,
        timesteps = 2000,
        sampling_timesteps = 250,
        objective = 'pred_noise',  # previous definition is "pred_v"
        clip_or_not = None,
        clip_range = None,

        force_ddim = False,
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = False,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion3D and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.conditional_diffusion = self.model.conditional_diffusion

        self.image_size_3D = image_size_3D

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        # clip to stablize
        self.clip_or_not = clip_or_not
        self.clip_range = clip_range
        assert self.clip_or_not is not None, 'clip_or_not must be specified'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs) # pre-defined schedule_fn_kwargs in main function arguments

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)  # This one is alpha(t-1)-bar. Practically, this pads the tensor with one element at the beginning and no elements at the end, using a padding value of 1.

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.force_ddim = force_ddim

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)  

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))  # this is sqrt(1- alpha(t)-bar)  /  sqrt(alpha(t)-bar) in the note

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # use option 2 in my note

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))  # The method a.clamp(min) applies element-wise clamping to a tensor min, ensuring that all values are greater than or equal to min
        register_buffer('posterior_mean_coef_x0', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))  # for x0
        register_buffer('posterior_mean_coef_xt', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))  # for xt

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)  # element-wise division
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    # @property is a built-in Python decorator that allows you to define a method as if it were a class attribute. This means you can access it like an attribute rather than calling it as a method.
    # for example, if in class "Circle" we have a function "area" as the property, then we can call this function as Circle.area instead of Circle.area()
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef_x0, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef_xt, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t,  condition = None, x_self_cond = None):
        if self.conditional_diffusion:
            if exists(condition) == 0:
                raise ValueError('conditional diffusion is specified, but no condition is provided')
            model_output = self.model(x, t, condition, x_self_cond)
        else:
            model_output = self.model(x, t, x_self_cond)

        if self.clip_or_not:
            maybe_clip = partial(torch.clamp, min = self.clip_range[0], max = self.clip_range[1]) 
        else:
            maybe_clip = identity
       

        if self.objective == 'pred_noise': 
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)

            x_start = maybe_clip(x_start)

            if self.clip_or_not:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)  # if prediction = ModelPrediction(pred_noise = a, x_start = b), then prediction.pred_noise = a, prediction.x_start = b

    def p_mean_variance(self, x, t, condition = None, x_self_cond = None, output_noise = True):
        if self.conditional_diffusion:
            if exists(condition) == 0:
                raise ValueError('conditional diffusion is specified, but no condition is provided')
            preds = self.model_predictions(x, t, condition, x_self_cond)
        else:
            preds = self.model_predictions(x, t, x_self_cond)
        
        x_start = preds.pred_x_start

        if self.clip_or_not:
            x_start.clamp_(self.clip_range[0], self.clip_range[1])
            # print('in p_mean_variance, x_start max and min: ',torch.max(x_start), torch.min(x_start))

        pred_noise = self.predict_noise_from_start(x, t, x_start)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        if output_noise == False:
            return model_mean, posterior_variance, posterior_log_variance, x_start
        else:
            return model_mean, posterior_variance, posterior_log_variance, x_start, pred_noise

    @torch.inference_mode()
    # In PyTorch, torch.inference_mode() is a context manager that temporarily sets the mode of the autograd engine to inference mode. This means that operations inside the context are treated as if they are being used for inference, rather than for training.
    def p_sample(self, x, t, condition = None, x_self_cond = None, output_noise = True):
        b, *_, device = *x.shape, self.device   # * in front of a list means unpacking the list, b = batch
        batched_times = torch.full((b,), t, device = device, dtype = torch.long) # torch.full() is a function in PyTorch that creates a tensor of a specified size and fills it with a specified value.
        
        if output_noise == False:
            model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, condition = condition, x_self_cond = x_self_cond, output_noise = output_noise)
        else:
            model_mean, _, model_log_variance, x_start, pred_noise = self.p_mean_variance(x = x, t = batched_times, condition = condition, x_self_cond = x_self_cond, output_noise = output_noise)
        
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise

        if output_noise == False:
            return pred_img, x_start
        else:
            return pred_img, x_start, pred_noise
        

    @torch.inference_mode()
    def p_sample_loop(self, shape, condition = None, return_all_timesteps = False, save_all_timesteps_ref = None, output_noise = True):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)  # this is random noise
        imgs = [img]

        x_start = None

        # gt = nb.load('/mnt/mount_zc_NAS/motion_correction/data/2D_motions/simulated_data_2D_spline/MO101701M000006/MO001A000007/static/image_data/recon_partial_ds.nii.gz')
        # gt_img = gt.get_fdata()[:,:,10:60]

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            if output_noise == True:
                img, x_start, pred_noise = self.p_sample(img, t, condition = condition, x_self_cond = self_cond)
            else:
                img, x_start = self.p_sample(img, t, condition = condition, x_self_cond = self_cond, output_noise = output_noise)

            imgs.append(img)
            
            if return_all_timesteps == True and ((t+1) % 50 == 0 or t == 0 or t == 1):
                img1 = torch.clone(x_start).detach().cpu().numpy().squeeze()
                img1 = Data_processing.crop_or_pad(img1, save_all_timesteps_ref[3], value = np.min(img1))
                img1 = Data_processing.normalize_image(img1, normalize_factor = save_all_timesteps_ref[2], image_max = save_all_timesteps_ref[1], image_min = save_all_timesteps_ref[0], invert = True)
                nb.save(nb.Nifti1Image(img1, save_all_timesteps_ref[4]), os.path.join(save_all_timesteps_ref[-1],  'time_' + str(t) + '_pred_x_start.nii.gz'))

        final_answer_x0 = img #if not return_all_timesteps else torch.stack(imgs, dim = 1)

        final_answer_x0 = self.unnormalize(final_answer_x0)
        return final_answer_x0

    @torch.inference_mode()
    def ddim_sample(self, shape, condition = None, return_all_timesteps = False, save_all_timesteps_ref = None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
    
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        coefficients = []
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            if condition is not None:
                pred = self.model_predictions(img, time_cond, condition, self_cond)
            else:
                pred = self.model_predictions(img, time_cond, self_cond)

            if time_next < 0:
                img = pred.pred_x_start
                imgs.append(img)
                
                if return_all_timesteps == True:
                    img1 = torch.clone(img).detach().cpu().numpy().squeeze()
                    img1 = Data_processing.crop_or_pad(img1, save_all_timesteps_ref[3], value = np.min(img1))
                    img1 = Data_processing.normalize_image(img1, normalize_factor = save_all_timesteps_ref[2], image_max = save_all_timesteps_ref[1], image_min = save_all_timesteps_ref[0], invert = True)
                    nb.save(nb.Nifti1Image(img1, save_all_timesteps_ref[4]), os.path.join(save_all_timesteps_ref[-1],  'time_' + str(time) + '_pred_x_start.nii.gz'))

                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]


            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = pred.pred_x_start * alpha_next.sqrt() + \
                  c * pred.pred_noise + \
                  sigma * noise

            if return_all_timesteps == True and ((time+1) % 50 == 0 or time == 0 or time == 1):
                img1 = torch.clone(pred.pred_x_start).detach().cpu().numpy().squeeze()
                img1 = Data_processing.crop_or_pad(img1, save_all_timesteps_ref[3], value = np.min(img1))
                img1 = Data_processing.normalize_image(img1, normalize_factor = save_all_timesteps_ref[2], image_max = save_all_timesteps_ref[1], image_min = save_all_timesteps_ref[0], invert = True)
                nb.save(nb.Nifti1Image(img1, save_all_timesteps_ref[4]), os.path.join(save_all_timesteps_ref[-1],  'time_' + str(time) + '_pred_x_start.nii.gz'))


            imgs.append(img)

        ret = img #if not return_all_timesteps else torch.stack(imgs, dim = 1)
        ret = self.unnormalize(ret)
        return ret


    @torch.inference_mode()
    def sample(self, condition = None, batch_size = 16, return_all_timesteps = False, save_all_timesteps_ref = None):
       
        if self.force_ddim == False:
            sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        else:
            sample_fn = self.ddim_sample
        return sample_fn((batch_size, self.channels, self.image_size_3D[0], self.image_size_3D[1], self.image_size_3D[2]), condition = condition, return_all_timesteps = return_all_timesteps, save_all_timesteps_ref = save_all_timesteps_ref)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        '''prepare random xt from x_start and t'''
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, condition = None, gt_for_mask = None, loss_weight_class = None, noise = None, offset_noise_strength = None):
        '''loss_weight_class is a list of [loss for bone, loss for brain, loss for air]'''
        b, c, h, w ,d = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                x_self_cond = self.model_predictions(x, t, condition).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        if self.conditional_diffusion:
            if exists(condition) == 0:
                raise ValueError('conditional diffusion is specified, but no condition is provided')
            model_out = self.model(x, t, condition, x_self_cond)
        else:
            model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        # MSE loss:
        loss = F.mse_loss(model_out, target, reduction = 'none')  #reduction='none' argument ensures that the loss is computed element-wise, without any reduction across batches.

        loss = reduce(loss, 'b ... -> b (...)', 'mean') # reduce() operates on the batch dimension (b) and potentially other dimensions (...). It reduces the loss tensor to have the same shape as the target tensor, with a mean reduction.
        loss = loss * extract(self.loss_weight, t, loss.shape)  # assign different loss weight to different timesteps
        return loss.mean()

        
    def forward(self, img, condition = None, gt_for_mask = None, loss_weight_class = None,  *args, **kwargs):
        b, c, h, w, d, device, img_size_3D, = *img.shape, img.device, self.image_size_3D
        # assert h == img_size_3D[0] and w == img_size_3D[1] and  d == img_size_3D[2], f'height and width and depta of image must be correct'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long() #torch.randint(a, b, size, device=None, dtype=None): This is a function in PyTorch used to generate random integers in a specified range. .long(): This method converts the tensor to have a data type of long, which is equivalent to 64-bit integer in PyTorch.

        # img = self.normalize(img)
        return self.p_losses(img, t, condition = condition, gt_for_mask = gt_for_mask, loss_weight_class = loss_weight_class, *args, **kwargs)
   

##### Trainer
class Trainer(object):
    def __init__(
        self,
        include_validation,
        diffusion_model,
        generator_train,
        train_batch_size,
        generator_val = None,

        train_num_steps = 100000, # total training epochs
        results_folder = None,
        train_lr = 1e-4,
        train_lr_decay_every = 200, 
        save_models_every = 1,
        validation_every = 1,
        
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),

        amp = False,
        mixed_precision_type = 'fp16',
        max_grad_norm = 1.,
         
    ):
        super().__init__()

        # accelerator
        self.accelerator = Accelerator(
            split_batches = True,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model
        self.model = diffusion_model   # it's not just the model architecture, but the actual model with loss calculation
        self.conditional_diffusion = self.model.conditional_diffusion
        self.channels = diffusion_model.channels

        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps

        self.image_size_3D = diffusion_model.image_size_3D

        # dataset and dataloader
        self.ds = generator_train
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())
        self.dl = self.accelerator.prepare(dl)
        self.cycle_dl = cycle(dl)

        self.validation = include_validation
        if self.validation:
            self.ds_val = generator_val
            dl_val = DataLoader(self.ds_val, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())
            self.dl_val = self.accelerator.prepare(dl_val)

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        self.scheduler = StepLR(self.opt, step_size = 1, gamma=0.95)
        self.max_grad_norm = max_grad_norm
        self.train_lr_decay_every = train_lr_decay_every
        self.save_model_every = save_models_every

        # for logging results in a folder periodically
        # EMA:
        # The purpose of using an EMA is to stabilize and improve the performance of a model during training. It achieves this by maintaining a smoothed version of the model's parameters, which reduces the impact of noise or fluctuations in the training process.
        #Typically, during training, you will update both the original model and the EMA model, but when you want to evaluate or make predictions, you would use the EMA model because it provides a more stable representation of the model's knowledge. This is especially useful in tasks like generative modeling, where you want to generate high-quality samples from the model.
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = results_folder
        ff.make_folder([self.results_folder])

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.validation_every = validation_every

    @property
    def device(self):
        return self.accelerator.device

    def save(self, stepNum):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'decay_steps': self.scheduler.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__}
        
        torch.save(data, os.path.join(self.results_folder, 'model-' + str(stepNum) + '.pt'))

    def load_model(self, trained_model_filename):
        print('load function!')
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(trained_model_filename, map_location=device)
        print('finish load')
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        self.scheduler.load_state_dict(data['decay_steps'])
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])


    def train(self, pre_trained_model = None ,start_step = None):
        accelerator = self.accelerator
        device = accelerator.device

        # load pre-trained
        if pre_trained_model is not None:
            self.load_model(pre_trained_model)
            print('model loaded from ', pre_trained_model)

        if start_step is not None:
            self.step = start_step
        
        self.scheduler.step_size = 1
        val_loss = np.inf
        training_log = []
        
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            
            while self.step < self.train_num_steps:
                print('training epoch: ', self.step + 1)
                print('learning rate: ', self.scheduler.get_last_lr()[0])

                average_loss = []
                count = 0
 
                for batch in self.dl:
                    self.opt.zero_grad()
                    if self.conditional_diffusion:
                        batch_x0, batch_condition = batch
                        data_x0 = batch_x0.to(device)
                        data_condition = batch_condition.to(device)
                        with self.accelerator.autocast():
                            loss = self.model(img = data_x0, condition = data_condition)
                    else:
                        data_x0 = batch.to(device)

                        with self.accelerator.autocast():
                            loss = self.model(img = data_x0)
                    
                    average_loss.append(loss.item())
                    count += 1
                    if count % 300 == 0:
                        print('batch count: ', count, 'loss: ', loss.item())

                    self.accelerator.backward(loss)
                    self.opt.step() 

                average_loss = sum(average_loss) / len(average_loss)
                pbar.set_description(f'average loss: {average_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.step += 1

                # save the model
                if self.step !=0 and divisible_by(self.step, self.save_model_every):
                   self.save(self.step)
                # update the parameter
                if self.step !=0 and divisible_by(self.step, self.train_lr_decay_every):
                    self.scheduler.step()

                self.ema.update()

                # do the validation if necessary
                if self.validation == True:
                    if self.step !=0 and divisible_by(self.step, self.validation_every):
                        print('validation at step: ', self.step)
                        self.model.eval()
                        with torch.no_grad():
                            val_loss = []
                            for batch in self.dl_val:
                                batch_x0, batch_condition = batch
                                data_x0 = batch_x0.to(device)
                                data_condition = batch_condition.to(device)
                                with self.accelerator.autocast():
                                    loss = self.model(img = data_x0, condition = data_condition)
                                
                                val_loss.append(loss.item())
                            val_loss = sum(val_loss) / len(val_loss)
                            print('validation loss: ', val_loss)
                        self.model.train(True)

                # save the training log
                training_log.append([self.step,average_loss, self.scheduler.get_last_lr()[0], val_loss])
                df = pd.DataFrame(training_log,columns = ['iteration','average_loss','learning_rate', 'validation_loss'])
                log_folder = os.path.join(os.path.dirname(self.results_folder),'log');ff.make_folder([log_folder])
                df.to_excel(os.path.join(log_folder, 'training_log.xlsx'),index=False)

                # at the end of each epoch, call on_epoch_end
                self.ds.on_epoch_end()
                if self.validation == True:
                    self.ds_val.on_epoch_end()
                pbar.update(1)

        accelerator.print('training complete')


# Sampling class
class Sampler(object):
    def __init__(
        self,
        diffusion_model,
        generator,
        batch_size,
        residual_learning, 
        device = 'cuda',
    ):
        super().__init__()

        # model
        self.model = diffusion_model  
        if device == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == 'cpu':
            self.device = torch.device("cpu")

        self.conditional_diffusion = self.model.conditional_diffusion
        self.channels = diffusion_model.channels
        self.image_size_3D = diffusion_model.image_size_3D
        self.batch_size = batch_size
        self.residual_learning = residual_learning

        # dataset and dataloader
        self.generator = generator
        dl = DataLoader(self.generator, batch_size = self.batch_size, shuffle = False, pin_memory = True, num_workers = 0)
        
        self.histogram_equalization = self.generator.histogram_equalization
        print('histogram equalization: ', self.histogram_equalization)
        self.bins = np.load('/mnt/camca_NAS/diffusion_ct_motion/data/histogram_equalization/bins.npy')
        self.bins_mapped = np.load('/mnt/camca_NAS/diffusion_ct_motion/data/histogram_equalization/bins_mapped.npy')         
        self.background_cutoff = self.generator.background_cutoff
        self.maximum_cutoff = self.generator.maximum_cutoff
        self.normalize_factor = self.generator.normalize_factor

        self.dl = dl
        self.cycle_dl = cycle(dl)
 
        # EMA:
        self.ema = EMA(diffusion_model)
        self.ema.to(self.device)

    def load_model(self, trained_model_filename):
        data = torch.load(trained_model_filename, map_location=self.device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.ema.load_state_dict(data["ema"])


    def sample_3D_slice_stack_w_trained_model(self, trained_model_filename, ground_truth_image_file, motion_image_file,  slice_range, save_file,save_gt_motion = None, not_start_from_first_slice = False):
        
        background_cutoff = self.background_cutoff; maximum_cutoff = self.maximum_cutoff; normalize_factor = self.normalize_factor
        self.load_model(trained_model_filename) 
        
        device = self.device

        self.ema.ema_model.eval()
        # check whether model is on GPU:
        print('model device: ', next(self.ema.ema_model.parameters()).device)

        gt = nb.load(ground_truth_image_file)
        gt_img = gt.get_fdata()
        if not_start_from_first_slice == False:
            gt_img = gt_img[:,:,slice_range[0]: slice_range[1]]
        else:
            gt_img = gt_img[:,:,slice_range[0] + 10: slice_range[1] + 10]
        gt_img = Data_processing.cutoff_intensity(gt_img, cutoff_low = background_cutoff, cutoff_high = maximum_cutoff)
        nb.save(nb.Nifti1Image(gt_img, gt.affine), os.path.join(os.path.dirname(save_file) , 'gt_slice' + str(slice_range[0]) +'to' + str(slice_range[1])+ '.nii.gz'))

        save_all_timesteps_ref = [background_cutoff, maximum_cutoff, normalize_factor, gt_img.shape, gt.affine, os.path.join(os.path.dirname(save_file), 'samples')]
        # ff.make_folder([os.path.join(os.path.dirname(save_file), 'samples')])

        # start to run
        with torch.inference_mode():
            if self.conditional_diffusion:
                datas = next(self.cycle_dl)
                data_condition = datas[1]
                print('data condition shape: ', data_condition.shape)
                data_condition_save = torch.clone(data_condition).numpy().squeeze() 
                data_condition_save = Data_processing.crop_or_pad(data_condition_save, [gt_img.shape[0], gt_img.shape[1], data_condition_save.shape[-1]], value = np.min(data_condition_save))
                data_condition_save = Data_processing.normalize_image(data_condition_save, normalize_factor = normalize_factor, image_max = maximum_cutoff, image_min = background_cutoff, invert = True)
                if self.histogram_equalization:
                    data_condition_save = Data_processing.apply_transfer_to_img(data_condition_save, self.bins, self.bins_mapped,reverse = True)
                nb.save(nb.Nifti1Image(data_condition_save, gt.affine), os.path.join(os.path.dirname(save_file), 'condition.nii.gz'))

                data_condition = data_condition.to(device)           
                        
                pred_img = self.ema.ema_model.sample(condition = data_condition, batch_size = self.batch_size ,return_all_timesteps = False, save_all_timesteps_ref = save_all_timesteps_ref )  
            else:
                pred_img = self.ema.ema_model.sample(batch_size = self.batch_size, return_all_timesteps = False, save_all_timesteps_ref = save_all_timesteps_ref)   
                         
        pred_img = pred_img.detach().cpu().numpy().squeeze()
        print(pred_img.shape)
    
        pred_img = Data_processing.crop_or_pad(pred_img, [gt_img.shape[0], gt_img.shape[1], self.image_size_3D[-1]], value = np.min(gt_img))
        pred_img = Data_processing.normalize_image(pred_img, normalize_factor = normalize_factor, image_max = maximum_cutoff, image_min = background_cutoff, invert = True)
        if self.histogram_equalization:
            pred_img = Data_processing.apply_transfer_to_img(pred_img, self.bins, self.bins_mapped,reverse = True)
        pred_img = Data_processing.correct_shift_caused_in_pad_crop_loop(pred_img)
      
        nb.save(nb.Nifti1Image(pred_img, gt.affine), save_file)


        # save gt and motion
        if save_gt_motion:
            if not_start_from_first_slice == False:
                motion_img = nb.load(motion_image_file).get_fdata()[:,:,slice_range[0]: slice_range[1]]
            else:
                motion_img = nb.load(motion_image_file).get_fdata()[:,:,slice_range[0] + 10: slice_range[1] + 10]
                
            motion_img = Data_processing.cutoff_intensity(motion_img, cutoff_low = background_cutoff, cutoff_high = maximum_cutoff)
            motion_img_save = nb.Nifti1Image(motion_img, gt.affine)
            nb.save(motion_img_save, os.path.join(os.path.dirname(save_file), 'motion_slice' + str(slice_range[0]) +'to' + str(slice_range[1])+ '.nii.gz'))
