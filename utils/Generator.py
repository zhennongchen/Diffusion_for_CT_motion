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

# histogram equalization pre-saved load
bins = np.load('/mnt/camca_NAS/diffusion_ct_motion/data/histogram_equalization/bins.npy')
bins_mapped = np.load('/mnt/camca_NAS/diffusion_ct_motion/data/histogram_equalization/bins_mapped.npy')

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
        img_list,
        condition_list,
        image_size_3D,

        patch_size,
        patch_stride, 
        original_patch_num,
        random_sampled_patch_num,
        patch_selection, 

        slice_number,
        slice_start,

        histogram_equalization,
        background_cutoff, 
        maximum_cutoff,
        normalize_factor,

        shuffle = False,
        augment = False,
        augment_frequency = 0,
    ):
        super().__init__()
        self.img_list = img_list
        self.condition_list = condition_list
        self.image_size_3D = image_size_3D
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.original_patch_num = original_patch_num
        self.random_sampled_patch_num = random_sampled_patch_num
        self.patch_selection = patch_selection
        self.slice_number = slice_number
        self.slice_start = slice_start
        self.histogram_equalization = histogram_equalization
        self.background_cutoff = background_cutoff
        self.maximum_cutoff = maximum_cutoff
        self.normalize_factor = normalize_factor
        self.shuffle = shuffle
        self.augment = augment
        self.augment_frequency = augment_frequency
        self.num_files = len(img_list)

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
        random_samples = ff.sample_patch_origins(self.original_patch_origins, self.random_sampled_patch_num , include_original_list = False)

        if self.shuffle == True:
            original_samples = [self.original_patch_origins[i] for i in random.sample(range(len(self.original_patch_origins)), self.original_patch_num)]
        else:
            original_samples = self.original_patch_origins[0:self.original_patch_num]
        
        self.final_patch_origins = original_samples + random_samples
       
        if self.patch_selection != None:
            print(self.patch_selection[0], self.patch_selection[1])
            self.final_patch_origins = self.final_patch_origins[self.patch_selection[0]: self.patch_selection[1]]
    

    def load_file(self, filename):
        ii = nb.load(filename).get_fdata()
    
        # do histogram equalization first
        if self.histogram_equalization == True:
            ii = Data_processing.apply_transfer_to_img(ii, bins, bins_mapped)
        # cutoff and normalization
        ii = Data_processing.cutoff_intensity(ii,cutoff_low = self.background_cutoff, cutoff_high = self.maximum_cutoff)
        ii = Data_processing.normalize_image(ii, normalize_factor = self.normalize_factor, image_max = self.maximum_cutoff, image_min = self.background_cutoff ,invert = False)

        return ii
        
    def __getitem__(self, index):
        # print('in this geiitem, self.index_array is: ', self.index_array)
        f,p = self.index_array[index]
        # print('index is: ', index, ' now we pick file ', f)
        x0_filename = self.img_list[f]
        # print('x0 filename is: ', x0_filename, ' while current x0 file is: ', self.current_x0_file)
        condition_file = self.condition_list[f]
        # print('condition file is: ', condition_file, ' while current condition file is: ', self.current_condition_file)

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
            # print('in this patient, the sampled patches are: ', self.final_patch_origins)
            
        # pick the slice range (can either be random or not fixed)
        if isinstance(self.slice_start, int): 
            self.slice_range = [self.slice_start, self.slice_start + self.slice_number]
        else:
            # self.slice_start is a range, given as [a,b], then please randomly pick a number in [a,b] including a and b
            count = 0
            while True:
                start = random.randint(self.slice_start[0], self.slice_start[1])
                self.slice_range = [start, start + self.slice_number]
                if self.slice_range[1] <= self.current_x0_data.shape[-1]- 4: # make some buffer so that the last slice of our cropped image is not the last slice of the entire image
                    break
                count += 1
                if count == 500:
                    start = 6; self.slice_range = [start, start + self.slice_number]; break
        # print('picked slice range: ', self.slice_range)

        x0_image_data = np.copy(self.current_x0_data)[:,:,self.slice_range[0]:self.slice_range[1]]

        x0_image_data = Data_processing.crop_or_pad(x0_image_data, [self.image_size_3D[0], self.image_size_3D[1], self.slice_number], value = np.min(x0_image_data))
        # print('in this getitem, image patch origin: ', self.final_patch_origins[p])
        x0_image_data = x0_image_data[self.final_patch_origins[p][0] : self.final_patch_origins[p][0] + self.patch_size, self.final_patch_origins[p][1] : self.final_patch_origins[p][1] + self.patch_size, ...]
        
        # print('shape of self.current_condition_data: ', self.current_condition_data.shape)
        condition_image_data = np.copy(self.current_condition_data)[:,: ,self.slice_range[0]:self.slice_range[1]]
        # print('shape of condition image data: ', condition_image_data.shape)
        condition_image_data = Data_processing.crop_or_pad(condition_image_data, [self.image_size_3D[0], self.image_size_3D[1], self.slice_number], value = np.min(condition_image_data))
        condition_image_data = condition_image_data[self.final_patch_origins[p][0] : self.final_patch_origins[p][0] + self.patch_size, self.final_patch_origins[p][1] : self.final_patch_origins[p][1] + self.patch_size, ...]

        # augmentation
        if self.augment == True:
            if random.uniform(0,1) < self.augment_frequency:
                # x0_image_data, z_rotate_degree = random_rotate(x0_image_data,  order = 0)
                x0_image_data, x_translate, y_translate = random_translate(x0_image_data)
                # condition_image_data, _ = random_rotate(condition_image_data, z_rotate_degree = z_rotate_degree, order = 0)
                condition_image_data, _, _ = random_translate(condition_image_data, x_translate = x_translate, y_translate = y_translate)
                # print('augment : z_rotate_degree, x_translate, y_translate: ', z_rotate_degree, x_translate, y_translate)
            
        x0_image_data = torch.from_numpy(x0_image_data).unsqueeze(0).float()
        condition_image_data = torch.from_numpy(condition_image_data).unsqueeze(0).float()

        # print('shape of x0 image data: ', x0_image_data.shape, ' and condition image data: ', condition_image_data.shape)
        return x0_image_data, condition_image_data
    
    def on_epoch_end(self):
        print('now run on_epoch_end function')
        self.index_array = self.generate_index_array()
        self.current_x0_file = None
        self.current_x0_data = None
        self.current_condition_file = None
        self.current_condition_data = None
      
