import sys 
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np
import Diffusion_models.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion_3D as ddpm_3D
import Diffusion_models.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_EDM as edm
import Diffusion_models.functions_collection as ff
import Diffusion_models.Defaults as Defaults
import Diffusion_models.Build_lists.Build_list as Build_list
import Diffusion_models.Generator as Generator

cg = Defaults.Parameters()

########################### set the trial name and pre-trained model path
trial_name = 'portable_EDM_patch_3Dmotion_hist_trial'
pre_trained_model = None # or path of the pre-trained model
start_step = 0 # if new training, start step = 0, if continue, start_step = None

########################### set the data path!
# define train
build_sheet =  Build_list.Build(os.path.join('/mnt/camca_NAS/diffusion_ct_motion/data/Patient_list/Patient_list_train_test_simulated_partial_motion_v2.xlsx'))  # this is data path for training data
_,_,_,_, _,_, x0_list1, _, condition_list1, _, _,_,_ = build_sheet.__build__(batch_list = [0,1,2,3])  # these are training batches
x0_list_train = np.copy(x0_list1); condition_list_train = np.copy(condition_list1)

print('train:', x0_list_train.shape, condition_list_train.shape)
print(x0_list_train[0:3], condition_list_train[0:3])

# set default, don't change unless necessary

image_size_3D = [256,256,50]
patch_size = 128
slice_number = 50; slice_start = [6,12] # if slice_start is an int then it will be the start slice, no random pick; if it is a range [a,  b], then randomly pick a starting slice in the range
val_slice_number = 20; val_slice_start = [20,21]

histogram_equalization = True
background_cutoff = -1000
maximum_cutoff = 2000
normalize_factor = 'equation'

# main code
model = ddpm_3D.Unet3D(
    init_dim = 64,
    channels = 1, 
    dim_mults = (1, 2, 4, 8),
    flash_attn = False,
    conditional_diffusion = True,
    full_attn = (None, None, False, True),
)
 
diffusion_model = edm.EDM(
    model,
    image_size = [patch_size, patch_size, image_size_3D[-1]],
    num_sample_steps = 50,
    clip_or_not = False,)

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


trainer = edm.Trainer(
    diffusion_model= diffusion_model,
    generator_train = generator_train,
    include_validation = False,
    train_batch_size = 1,

    train_num_steps = 10000, # total training epochs
    results_folder = os.path.join('/mnt/camca_NAS/diffusion_ct_motion/models', trial_name, 'models'),
   
    train_lr = 1e-4,
    train_lr_decay_every = 100, 
    save_models_every = 1,
    validation_every = None,
)

trainer.train(pre_trained_model=pre_trained_model, start_step= start_step)