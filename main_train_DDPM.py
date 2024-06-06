'''the code is specifically for training the DDPM model on the cases collected on 202404, but can be easily adapted for other cases by changing the data path'''
'''run this code by python main_train_DDPM.py'''
import sys 
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np
import Diffusion_for_CT_motion.diffusion_models.conditional_DDPM_3D as ddpm_3D
import Diffusion_for_CT_motion.utils.functions_collection as ff
import Diffusion_for_CT_motion.Build_lists.Build_list as Build_list
import Diffusion_for_CT_motion.utils.Generator as Generator

########################### set the trial name and pre-trained model path
trial_name = 'portable_DDPM_patch_3Dmotion_hist_v1'
pre_trained_model = None # or path of the pre-trained model
start_step = 0 # if new training, start step = 0, if continue, start_step = None

########################### set the data path!
# define train
build_sheet =  Build_list.Build(os.path.join('/mnt/camca_NAS/diffusion_ct_motion/data/Patient_list/Patient_list_train_test_simulated_all_motion_v1.xlsx'))  # this is data path for training data
_,_,_,_, _,_, x0_list1, _, condition_list1, _, _,_,_ = build_sheet.__build__(batch_list = [0,1,2,3]) 
x0_list_train = np.copy(x0_list1); condition_list_train = np.copy(condition_list1)

print('train:', x0_list_train.shape, condition_list_train.shape)
print(x0_list_train[0:3], condition_list_train[0:3])


# set default, don't change unless necessary
image_size_3D = [256,256,50] 
patch_size = 128 
slice_number = 50; slice_start = [6,12] 
val_slice_number = 20; val_slice_start = [20,21]
objective = 'pred_x0'
timesteps = 1000

histogram_equalization = True
background_cutoff = -500
maximum_cutoff =  None
normalize_factor = 1000


# define u-net and diffusion model
model = ddpm_3D.Unet3D(
    init_dim = 64,
    channels = 1, 
    dim_mults = (1, 2, 4, 8),
    flash_attn = False,
    conditional_diffusion = True,
    full_attn = (None, None, False, True),)

diffusion_model = ddpm_3D.GaussianDiffusion3D(
    model,
    image_size_3D = [patch_size, patch_size, image_size_3D[-1]],
    timesteps = timesteps,          
    auto_normalize=False,
    objective = objective,
    clip_or_not = False,
)

# generator definition
generator_train = Generator.Dataset_dual_patch(
    x0_list_train,
    condition_list_train,
    image_size_3D = image_size_3D,
    patch_size = patch_size,
    patch_stride = patch_size,
    original_patch_num = 1,
    random_sampled_patch_num = 2,
    patch_selection = None,
    slice_number = slice_number,
    slice_start = slice_start,

    histogram_equalization = histogram_equalization, 
    background_cutoff = background_cutoff, 
    maximum_cutoff = maximum_cutoff,
    normalize_factor = normalize_factor,
    shuffle = True,
    augment = True,  # only translation
    augment_frequency = 0.2,)

# not include validation in this script
# generator_val = Generator.Dataset_dual_patch(
#     x0_list_val,
#     condition_list_val,
#     image_size_3D = [image_size_3D[0], image_size_3D[1], val_slice_number],
#     patch_size = 256,
#     patch_stride = 1,
#     original_patch_num = 1,
#     random_sampled_patch_num = 0,
#     patch_selection = None,
#     slice_number = val_slice_number,
#     slice_start = val_slice_start,

#     histogram_equalization = histogram_equalization, 
#     background_cutoff = background_cutoff, 
#     maximum_cutoff = maximum_cutoff,
#     normalize_factor = normalize_factor,)


# start to train
trainer = ddpm_3D.Trainer(
    diffusion_model= diffusion_model,
    generator_train = generator_train,
    include_validation = False, # no validation in this script
    generator_val = None,
    train_batch_size = 1,

    train_num_steps = 5000, # total training epochs
    results_folder = os.path.join('/mnt/camca_NAS/diffusion_ct_motion','models', trial_name, 'models'),
   
    train_lr = 1e-4,
    train_lr_decay_every = 100, 
    save_models_every = 1,
    validation_every = None,)


trainer.train(pre_trained_model = pre_trained_model, start_step= start_step )