import numpy as np
import glob 
import os
from PIL import Image
import math 
import SimpleITK as sitk
import cv2
import random
import nibabel as nb
from dipy.align.reslice import reslice
import Diffusion_models.Data_processing as dp
from skimage.metrics import structural_similarity as compare_ssim

# function: histogram equalization
def equalize_histogram(bins, hist, weight):
    '''
    Equalize the histogram such that the cumulative distribution function is linear.
    '''
    # normalized cdf
    cdf = np.cumsum(hist) / np.sum(hist)

    # target cdf
    cdf_target = np.linspace(0, 1, len(cdf))

    bins_mapped = np.interp(cdf, cdf_target, bins)

    # weight the original and the mapped bins
    bins_mapped = weight * bins_mapped + (1 - weight) * bins

    return bins_mapped

def apply_transfer_to_img(img: np.array, bins: np.array, bins_mapped: np.array, reverse=False):
    '''
    Apply the transfer function to the image.
    The value outside the transfer range should be preserved.
    '''
    if reverse:
        bins, bins_mapped = bins_mapped, bins

    mask = (img > bins[0]) & (img < bins[-1])
    img_mapped = np.interp(img.astype(np.float32), bins, bins_mapped)
    img_mapped[~mask] = img[~mask]

    return img_mapped

# function: set window level
def set_window(image,level,width):
    if len(image.shape) == 3:
        image = image.reshape(image.shape[0],image.shape[1])
    new = np.copy(image)
    high = level + width // 2
    low = level - width // 2
    # normalize
    unit = (1-0) / (width)
    new[new>high] = high
    new[new<low] = low
    new = (new - low) * unit 
    return new


# function: find all files under the name * in the main folder, put them into a file list
def find_all_target_files(target_file_name,main_folder):
    F = np.array([])
    for i in target_file_name:
        f = np.array(sorted(glob.glob(os.path.join(main_folder, os.path.normpath(i)))))
        F = np.concatenate((F,f))
    return F

# function: find time frame of a file
def find_timeframe(file,num_of_dots,start_signal = '/',end_signal = '.'):
    k = list(file)

    if num_of_dots == 0: 
        num = [i for i,e in enumerate(k) if e== start_signal][-1]
        kk = k[num+1:]
    
    else:
        if num_of_dots == 1: #.png
            num1 = [i for i, e in enumerate(k) if e == end_signal][-1]
        elif num_of_dots == 2: #.nii.gz
            num1 = [i for i, e in enumerate(k) if e == end_signal][-2]
        num2 = [i for i,e in enumerate(k) if e== start_signal][-1]
        kk=k[num2+1:num1]
    total = 0
    for i in range(0,len(kk)):
        total += int(kk[i]) * (10 ** (len(kk) - 1 -i))
    return total

# function: sort files based on their time frames
def sort_timeframe(files,num_of_dots,start_signal = '/',end_signal = '.'):
    time=[]
    time_s=[]
    
    for i in files:
        a = find_timeframe(i,num_of_dots,start_signal,end_signal)
        time.append(a)
        time_s.append(a)
    time_s.sort()
    new_files=[]
    for i in range(0,len(time_s)):
        j = time.index(time_s[i])
        new_files.append(files[j])
    new_files = np.asarray(new_files)
    return new_files

# function: make folders
def make_folder(folder_list):
    for i in folder_list:
        os.makedirs(i,exist_ok = True)



# function: patch definition:
def patch_definition(img_shape, patch_size, stride, count_for_overlap = False):
    # now assume patch_size is square in x and y, and same dimension as img in z 
    start = 0
    end = img_shape[0] - patch_size

    origin_x_list = np.arange(start, end+1, stride)

    patch_origin_list = [[origin_x_list[i], origin_x_list[j]] for i in range(0,origin_x_list.shape[0]) for j in range(0,origin_x_list.shape[0])]
    
    if count_for_overlap == False:
        return patch_origin_list, 0
    else:
        count = np.zeros(img_shape)
        for origin in patch_origin_list:
            count[origin[0] : (origin[0] + patch_size), origin[1] : (origin[1] + patch_size), :] += 1

        return patch_origin_list, count

# function: randomly sample patch origins
def sample_patch_origins(patch_origins, N, include_original_list = None):
    if isinstance(patch_origins, list) == False:
        patch_origins = patch_origins.tolist()
    origins = np.array(patch_origins)
    x_min,  x_max, y_min, y_max = np.min(origins[:,0]), np.max(origins[:,0]), np.min(origins[:,1]), np.max(origins[:,1])

    # Generate random coordinates
    pixels = [(random.randint(x_min, x_max), random.randint(y_min, y_max)) for _ in range(N)]

    if include_original_list is True:
        pixels = patch_origins + pixels

    return pixels

# function: generate angle list
def get_angles_zc(nview, total_angle,start_angle):
    return np.arange(0, nview, dtype=np.float32) * (total_angle / 180 * np.pi) / nview + (start_angle / 180 * np.pi)

# function: resample nii files
def resample_nifti(nifti, 
                   order,
                   mode, #'nearest' or 'constant' or 'reflect' or 'wrap'    
                   cval,
                   in_plane_resolution_mm=1.25,
                   slice_thickness_mm=None,
                   number_of_slices=None):
    
    # sometimes dicom to nifti programs don't define affine correctly.
    resolution = np.array(nifti.header.get_zooms()[:3] + (1,))
    if (np.abs(nifti.affine)==np.identity(4)).all():
        nifti.set_sform(nifti.affine*resolution)


    data   = nifti.get_fdata().copy()
    shape  = nifti.shape[:3]
    affine = nifti.affine.copy()
    zooms  = nifti.header.get_zooms()[:3] 

    if number_of_slices is not None:
        new_zooms = (in_plane_resolution_mm,
                     in_plane_resolution_mm,
                     (zooms[2] * shape[2]) / number_of_slices)
    elif slice_thickness_mm is not None:
        new_zooms = (in_plane_resolution_mm,
                     in_plane_resolution_mm,
                     slice_thickness_mm)            
    else:
        new_zooms = (in_plane_resolution_mm,
                     in_plane_resolution_mm,
                     zooms[2])

    new_zooms = np.array(new_zooms)
    for i, (n_i, res_i, res_new_i) in enumerate(zip(shape, zooms, new_zooms)):
        n_new_i = (n_i * res_i) / res_new_i
        # to avoid rounding ambiguities
        if (n_new_i  % 1) == 0.5: 
            new_zooms[i] -= 0.001

    data_resampled, affine_resampled = reslice(data, affine, zooms, new_zooms, order=order, mode=mode , cval = cval)
    nifti_resampled = nb.Nifti1Image(data_resampled, affine_resampled)

    x=nifti_resampled.header.get_zooms()[:3]
    y=new_zooms
    if not np.allclose(x,y, rtol=1e-02):
        print('not all close: ', x,y)

    return nifti_resampled   

    
