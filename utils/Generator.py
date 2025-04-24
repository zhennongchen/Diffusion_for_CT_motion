'''this script is the data generator (to output patches for model input and model condition)'''
import os
import numpy as np
import nibabel as nb
import random
from scipy import ndimage
from skimage.measure import block_reduce

import torch
from torch.utils.data import Dataset
import Diffusion_for_CT_motion.utils.Data_processing as Data_processing
import Diffusion_for_CT_motion.utils.functions_collection as ff

# random function
def random_rotate(i, z_rotate_degree = None, z_rotate_range = [0,0], fill_val = None, order = 0):
    # only do rotate according to z (in-plane rotation)
    if z_rotate_degree is None:
        z_rotate_degree = random.uniform(z_rotate_range[0], z_rotate_range[1])

    if fill_val is None:
        fill_val = np.min(i)
    
    if z_rotate_degree == 0:
        return i, z_rotate_degree
    else:
        return Data_processing.rotate_image(np.copy(i), [0,0,z_rotate_degree], order = order, fill_val = fill_val, ), z_rotate_degree

def random_translate(i, x_translate = None,  y_translate = None, translate_range = [-10,10]):
    # only do translate according to x and y
    if x_translate is None or y_translate is None:
        x_translate = int(random.uniform(translate_range[0], translate_range[1]))
        y_translate = int(random.uniform(translate_range[0], translate_range[1]))

    return Data_processing.translate_image(np.copy(i), [x_translate,y_translate,0]), x_translate,y_translate


class Dataset_dual_patch(Dataset):
    def __init__(
        self,
        x0_list,  # this is the x0 in the diffusion framework, which is the image we want to generate --> motion-free image
        condition_list, # this is the motion-corrupted image

        image_size_3D,
        slice_start,

        # for 3D-patch-wise training
        patch_size,
        patch_stride,  # equal to patch size
        original_patch_num, # original_patch means segment the entire image into non-overlapped patches with set dimensions. for example, if we have a 256x256 image and the patch size is 128x128, then we can segment the image into 4 patches.
        random_sampled_patch_num, # random_sampled_patch means randomly sample patches from the original image, the vertex of the patchs can be anywhere in the image. 
        
        # for histogram equalization, an important image preprocessing step in our pipeline
        histogram_equalization,
        bins,
        bins_mapped,

        # for data normalization
        background_cutoff, 
        maximum_cutoff,
        normalize_factor,

        # for augmentation
        shuffle = False,
        augment = False,
        augment_frequency = 0,
    ):
        super().__init__()
        self.x0_list = x0_list
        self.condition_list = condition_list
        self.image_size_3D = image_size_3D
        self.slice_start = slice_start
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.original_patch_num = original_patch_num
        self.random_sampled_patch_num = random_sampled_patch_num

        self.histogram_equalization = histogram_equalization
        self.bins = bins
        self.bins_mapped = bins_mapped
        self.background_cutoff = background_cutoff
        self.maximum_cutoff = maximum_cutoff
        self.normalize_factor = normalize_factor
        self.shuffle = shuffle
        self.augment = augment
        self.augment_frequency = augment_frequency
        self.num_files = len(x0_list)

        self.original_patch_origins, _ = ff.patch_definition(self.image_size_3D, self.patch_size, self.patch_stride)
        assert self.original_patch_num <= len(self.original_patch_origins)

        # final patch num
        self.patch_num = self.original_patch_num + self.random_sampled_patch_num

        self.index_array = self.generate_index_array()
        self.current_x0_file = None
        self.current_x0_data = None
        self.current_condition_file = None
        self.current_condition_data = None
       

    def generate_index_array(self):
        np.random.seed()
        index_array = []
        
        if self.shuffle == True:
            f_list = np.random.permutation(self.num_files)
        else:
            f_list = np.arange(self.num_files)

        for f in f_list:
            if self.shuffle == True:
                p_list = np.random.permutation(self.patch_num)
            else:
                p_list = np.arange(self.patch_num)
            for p in p_list:
                index_array.append([f, p])
        return index_array

    def __len__(self):
        return self.num_files * self.patch_num

    def sample_patches(self):
        # original patches
        if self.shuffle == True:
            original_samples = [self.original_patch_origins[i] for i in random.sample(range(len(self.original_patch_origins)), self.original_patch_num)]
        else:
            original_samples = self.original_patch_origins[0:self.original_patch_num]

        # random patches
        random_samples = ff.sample_patch_origins(self.original_patch_origins, self.random_sampled_patch_num , include_original_list = False)
        
        self.final_patch_origins = original_samples + random_samples
       
    def load_file(self, filename):
        ii = nb.load(filename).get_fdata()
    
        # do histogram equalization first
        if self.histogram_equalization == True:
            ii = Data_processing.apply_transfer_to_img(ii, self.bins, self.bins_mapped)
        # cutoff and normalization
        ii = Data_processing.cutoff_intensity(ii,cutoff_low = self.background_cutoff, cutoff_high = self.maximum_cutoff)
        ii = Data_processing.normalize_image(ii, normalize_factor = self.normalize_factor, image_max = self.maximum_cutoff, image_min = self.background_cutoff ,invert = False)
        return ii
        
    def __getitem__(self, index):
        f,p = self.index_array[index] # f is the index of the file, p is the index of the patch
        x0_filename = self.x0_list[f]
        condition_file = self.condition_list[f]

        if x0_filename != self.current_x0_file:
            x0_img = self.load_file(x0_filename)
            self.current_x0_file = x0_filename
            self.current_x0_data = np.copy(x0_img)

        if condition_file != self.current_condition_file:
            # print('it is a new case, load the file')
            condition_img = self.load_file(condition_file)
            self.current_condition_file = condition_file
            self.current_condition_data = np.copy(condition_img)

            # sample patches for this case:
            self.sample_patches()
            
        # pick the slice range (when 3D data has more than 50 slices --> our model takes [x,y,50] for trianing)
        if isinstance(self.slice_start, int):  # if slice_start is an int then it will be the start slice, no random pick
            self.slice_range = [self.slice_start, self.slice_start + self.image_size_3D[-1]]
        else: # slice_start is a range, given as [a,b], then please randomly pick a number in [a,b] including a and b
            while True:
                start = random.randint(self.slice_start[0], self.slice_start[1])
                self.slice_range = [start, start + self.image_size_3D[-1]]
                if self.slice_range[1] <= self.current_x0_data.shape[-1]- 2:
                    break

        # make x0 and condition ready
        x0_image_data = np.copy(self.current_x0_data)[:,:,self.slice_range[0]:self.slice_range[1]]
        x0_image_data = Data_processing.crop_or_pad(x0_image_data, [self.image_size_3D[0], self.image_size_3D[1], self.image_size_3D[2]], value = np.min(x0_image_data))
        x0_image_data = x0_image_data[self.final_patch_origins[p][0] : self.final_patch_origins[p][0] + self.patch_size, self.final_patch_origins[p][1] : self.final_patch_origins[p][1] + self.patch_size, ...]

        condition_image_data = np.copy(self.current_condition_data)[:,: ,self.slice_range[0]:self.slice_range[1]]
        condition_image_data = Data_processing.crop_or_pad(condition_image_data, [self.image_size_3D[0], self.image_size_3D[1], self.image_size_3D[2]], value = np.min(condition_image_data))
        condition_image_data = condition_image_data[self.final_patch_origins[p][0] : self.final_patch_origins[p][0] + self.patch_size, self.final_patch_origins[p][1] : self.final_patch_origins[p][1] + self.patch_size, ...]

        # augmentation
        if self.augment == True:
            if random.uniform(0,1) < self.augment_frequency:
                # x0_image_data, z_rotate_degree = random_rotate(x0_image_data,  order = 1)
                x0_image_data, x_translate, y_translate = random_translate(x0_image_data)
                # condition_image_data, _ = random_rotate(condition_image_data, z_rotate_degree = z_rotate_degree, order = 1)
                condition_image_data, _, _ = random_translate(condition_image_data, x_translate = x_translate, y_translate = y_translate)
                # print('augment : z_rotate_degree, x_translate, y_translate: ', z_rotate_degree, x_translate, y_translate)
            
        x0_image_data = torch.from_numpy(x0_image_data).unsqueeze(0).float()
        condition_image_data = torch.from_numpy(condition_image_data).unsqueeze(0).float()
        return x0_image_data, condition_image_data
    
    def on_epoch_end(self):
        print('now run on_epoch_end function')
        self.index_array = self.generate_index_array()
        self.current_x0_file = None
        self.current_x0_data = None
        self.current_condition_file = None
        self.current_condition_data = None
      
