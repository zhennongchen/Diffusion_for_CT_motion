import numpy as np
import glob 
import os
from PIL import Image
import math
import SimpleITK as sitk
import nibabel as nb
from dipy.align.reslice import reslice
import Diffusion_for_CT_motion.simulation_data_generation.Data_processing as dp
import CTProjector.src.ct_projector.projector.numpy as ct_projector

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
def txt_writer(save_path, replace , parameters,names):
    if replace == True:
        t_file = open(save_path,"w+")
    else:
        t_file = open(save_path,"a")
 
    for i in range(0,len(parameters)):
        t_file.write(names[i] + ': ')
        for ii in range(0,len(parameters[i])):
            t_file.write(str(np.round(parameters[i][ii],3))+' ')
        t_file.write('\n')
    t_file.write('\n\n')
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
    # ssim = compare_ssim(a,b)

    # # normalized mean squared error
    # nmse = np.mean((a-b)**2) / mean_square_value

    # # normalized root mean squared error
    # nrmse = rmse / mean_square_value

    # # peak signal-to-noise ratio
    # psnr = 10 * (math.log10((8191**2) / mse ))

    return mae, mse, rmse, r_rmse, ssim


# function: random augmentation parameters
def augment_parameters_2D(dim = [256,256], percent = 0.04, frequency = 0.5):
    do_or_not = np.random.rand()
    if do_or_not > (1-frequency):
        do = 1
        augment_tx = np.random.rand() * dim[0] * percent; augment_tx = [augment_tx if np.random.rand() > 0.5 else -augment_tx for i in range(0,1)][0]
        augment_ty = np.random.rand() * dim[1] * percent; augment_ty = [augment_ty if np.random.rand() > 0.5 else -augment_ty for i in range(0,1)][0]
        augment_r = np.random.rand() * 5; augment_r = [augment_r if np.random.rand() > 0.5 else -augment_r for i in range(0,1)][0]
        return do, augment_tx, augment_ty, augment_r
    else:
        return 0,0,0,0

## function: optimize
def optimize(pred, true, num = 4, mode = [0,1,2], rank_max = 3, random_rank = True,  boundary = 1.5):
    true = np.reshape(true, -1)
    assert pred.shape[1] == true.shape[0]

    final_answer_list = []

    # average
    final_answer_average = np.zeros(num)
    for j in range(0,pred.shape[1]):
        col = pred[:,j]
        col_rank = np.sort(col)
        # if col.shape[0] > 3:
        #     # remove lowest and highest
        #     if abs(col_rank[0] - col_rank[1]) >= 1:
        #         col_rank = np.delete(col_rank,0)
        #     if abs(col_rank[-1] - col_rank[-2]) >= 1:
        #         col_rank = np.delete(col_rank,-1)
        final_answer_average[j] = np.mean(col_rank)
    
    if 0 in mode:
        final_answer_list.append(final_answer_average)

    # pick best model (w/lowest MAE)
    errors = np.mean(np.abs(pred - true), axis=1)
    min_index = np.argmin(errors)
    final_answer_model_mae = pred[min_index]
    if 1 in mode:
        final_answer_list.append(final_answer_model_mae)
    
    # pick best model (w/lowest largest difference)
    errors = np.max(np.abs(pred - true), axis=1)
    min_index = np.argmin(errors)
    final_answer_model_diff = pred[min_index]
    if 2 in mode:
        return final_answer_model_diff
        # final_answer_list.append(final_answer_model_diff)

    # pick element-wise
    mae = np.abs(pred - true)
    # Find the index of the element with the smallest MAE in each column
    index = np.argmin(mae, axis=0)
    
    # Select the corresponding elements from each column of "pred"
    final_answer_model_element = pred[index, np.arange(pred.shape[1])]
    if 3 in mode:
        return final_answer_model_element
    
    final_answer_list = np.reshape(np.asarray(final_answer_list), (-1, num))
    errors = np.mean(np.abs(final_answer_list - true), axis=1)
    min_index = np.argmin(errors)
   
    return final_answer_list[min_index]




# function: round difference: if abs(a-b)  % 1 <= threshold, then a-b = math.floor(a-b)
def round_diff(pred, gt, threshold):
    # b is ground truth
    A = pred - gt 
    rounded_A = np.where((np.abs(A) % 1 <= threshold) & (A < 0), np.ceil(A), np.where((np.abs(A) % 1 <= threshold) & (A > 0), np.floor(A), A))
    return gt + rounded_A

# function: read real-scan projection data
def read_projection_data(
    input_dir, projector: ct_projector.ct_projector, start_view, end_view, nrot_per_slab, nz_per_slice
):
    min_file_bytes = 1024 * 10  # file size should be at least 10MB

    if end_view < 0:
        end_view = len(find_all_target_files(['slab_*.nii.gz'], input_dir)) -1 
       
    filenames = []
    for iview in range(start_view, end_view + 1):
        filename = os.path.join(input_dir, f'slab_{iview}.nii.gz')
        if not os.path.exists(filename):
            break
        if os.path.getsize(filename) < min_file_bytes:
            break
        filenames.append(filename)

    prjs = []
    print('Reading data from {0} files'.format(len(filenames)), flush=True)
    for i, filename in enumerate(filenames):
        print(i, end=',', flush=True)
        prj = sitk.GetArrayFromImage(sitk.ReadImage(filename))
        prj = prj.astype(np.float32)
        # print('shape: ', prj.shape)
        prjs.append(prj)
    prjs = np.array(prjs)

    prjs = prjs.reshape([
        nrot_per_slab,
        prjs.shape[0] // nrot_per_slab,
        prjs.shape[1],
        prjs.shape[2] // nz_per_slice,
        nz_per_slice,
        prjs.shape[3],
    ])

    print('prjs shape ',prjs.shape)

    prjs = np.mean(prjs, axis=(0, 4))

    print('prjs shape ',prjs.shape)

    # update projector
    projector.nv = prjs.shape[2]
    projector.nz = prjs.shape[2]
    projector.dv = projector.dv * nz_per_slice
    projector.dz = projector.dv

    return prjs, projector

# function: hanning filter
def hann_filter(x, projector):
    x_prime = np.fft.fft(x)
    x_prime = np.fft.fftshift(x_prime)
    hanning_window = np.hanning(projector.nu)
    x_prime_hann = x_prime * hanning_window
    x_inverse_hann = np.fft.ifft(np.fft.ifftshift(x_prime_hann))
    return x_inverse_hann

def apply_hann(prjs, projector):
    prjs_hann = np.zeros_like(prjs)
    for ii in range(0,prjs_hann.shape[0]):
        for jj in range(0, prjs_hann.shape[2]):
            for kk in range(0, prjs_hann.shape[1]):
                prjs_hann[ii,kk,jj,:] = hann_filter(prjs[ii,kk,jj,:], projector)
    return prjs_hann



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
    
