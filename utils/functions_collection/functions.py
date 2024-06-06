import numpy as np
import glob 
import os
from PIL import Image
import math
import SimpleITK as sitk
import cv2
import random
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
    

# function: generate angle list
def get_angles_zc(nview, total_angle,start_angle):
    return np.arange(0, nview, dtype=np.float32) * (total_angle / 180 * np.pi) / nview + (start_angle / 180 * np.pi)

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

# function: get first X numbers
# if we have 1000 numbers, how to get the X number of every interval numbers?
def get_X_numbers_in_interval(total_number, start_number, end_number , interval = 100):
    '''if no random pick, then random_pick = [False,0]; else, random_pick = [True, X]'''
    n = []
    for i in range(0, total_number, interval):
        n += [i + a for a in range(start_number,end_number)]
    n = np.asarray(n)
    return n


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

# function: write txt file
def txt_writer(save_path,parameters,names):
    t_file = open(save_path,"w+")
    for i in range(0,len(parameters)):
        t_file.write(names[i] + ': ')
        for ii in range(0,len(parameters[i])):
            t_file.write(str(np.round(parameters[i][ii],3))+' ')
        t_file.write('\n')
    t_file.close()



# function: insert blank slices (to make FOV larger than image object)
def insert_blank_slices(img,insert_to_which_direction , begin_blank_slice_num = 10, end_blank_slice_num = 10):
    
    min_val = np.min(img)

    if insert_to_which_direction == 'x':
        new_img = np.zeros([(img.shape[0] + begin_blank_slice_num + end_blank_slice_num), img.shape[1], img.shape[2]]) + min_val
        new_img[begin_blank_slice_num :begin_blank_slice_num + img.shape[0],...] = img

    if insert_to_which_direction == 'y':
        new_img = np.zeros([img.shape[0], (img.shape[1] + begin_blank_slice_num + end_blank_slice_num), img.shape[2]]) + min_val
        new_img[:, begin_blank_slice_num :begin_blank_slice_num + img.shape[1], :] = img

    if insert_to_which_direction == 'z':
        new_img = np.zeros([img.shape[0], img.shape[1], (img.shape[2] + begin_blank_slice_num + end_blank_slice_num)]) + min_val
        new_img[:, :, begin_blank_slice_num :begin_blank_slice_num + img.shape[2]] = img
    
    return new_img


# function: save grayscale image
def save_grayscale_image(a,save_path,normalize = True, WL = 50, WW = 100):
    I = np.zeros((a.shape[0],a.shape[1],3))
    # normalize
    if normalize == True:
        a = set_window(a, WL, WW)

    for i in range(0,3):
        I[:,:,i] = a
    
    Image.fromarray((I*255).astype('uint8')).save(save_path)


# function: normalize translation control points:
def convert_translation_control_points(t, dim, from_pixel_to_1 = True):
    if from_pixel_to_1 == True: # convert to a space -1 ~ 1
        t = [tt / dim * 2 for tt in t]
    else: # backwards
        t = [tt / 2 * dim for tt in t]
    
    return np.asarray(t)


# function: comparison error
def compare(a, b,  cutoff_low = 0 ,cutoff_high = 1000000, extreme = 5000):
    # compare a to b, b is ground truth
    # if a pixel is lower than cutoff (meaning it's background), then it's out of comparison
    c = np.copy(b)
    diff = abs(a-b)
   
    a = a[(c>cutoff_low)& (c < cutoff_high) & (diff<extreme)].reshape(-1)
    b = b[(c>cutoff_low)& (c < cutoff_high) & (diff<extreme)].reshape(-1)

    diff = abs(a-b)

    # mean absolute error
    mae = np.mean(abs(a - b)) 

    # mean squared error
    mse = np.mean((a-b)**2) 

    # root mean squared error
    rmse = math.sqrt(mse)

    # relative root mean squared error
    dominator = math.sqrt(np.mean(b ** 2))
    r_rmse = rmse / dominator * 100

    # structural similarity index metric
    cov = np.cov(a,b)[0,1]
    ssim = (2 * np.mean(a) * np.mean(b)) * (2 * cov) / (np.mean(a) ** 2 + np.mean(b) ** 2) / (np.std(a) ** 2 + np.std(b) ** 2)

    return mae, mse, rmse, r_rmse, ssim

# function: dice
def np_categorical_dice(pred, truth, k):
    """ Dice overlap metric for label k """
    A = (pred == k).astype(np.float32)
    B = (truth == k).astype(np.float32)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B))


# function: erode and dilate
def erode_and_dilate(img_binary, kernel_size, erode = None, dilate = None):
    img_binary = img_binary.astype(np.uint8)

    kernel = np.ones(kernel_size, np.uint8)  

    if dilate is True:
        img_binary = cv2.dilate(img_binary, kernel, iterations = 1)

    if erode is True:
        img_binary = cv2.erode(img_binary, kernel, iterations = 1)
    return img_binary


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

    
