import numpy as np
import nibabel as nb
import os
from skimage.measure import block_reduce
from scipy import ndimage
from dipy.align.reslice import reslice
import Diffusion_for_CT_motion.utils.functions_collection as ff

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


def crop_or_pad(array, target, value):
    # Pad each axis to at least the target.

    margin = target - np.array(array.shape)
    padding = [(0, max(x, 0)) for x in margin]
    array = np.pad(array, padding, mode="constant", constant_values=value)
    for i, x in enumerate(margin):
        array = np.roll(array, shift=+(x // 2), axis=i)

    if type(target) == int:
        target = [target] * array.ndim

    ind = tuple([slice(0, t) for t in target])
    return array[ind]

def correct_shift_caused_in_pad_crop_loop(img):
    # if an image goes from [a,b,c] --> pad --> [A,B,c] --> crop --> [a,b,c], when a,b is even, it goes back to original image, but when a,b is odd, it need to shift by 1 pixel in x and y
    if img.shape[0] % 2 == 1:

        img = np.roll(img, shift = 1, axis = 0)
        img = np.roll(img, shift = 1, axis = 1)
    else:
        img = np.copy(img)
    return img


def adapt(x, cutoff = False,add_noise = False, sigma = 5, normalize = True, expand_dim = True):
    x = np.load(x, allow_pickle = True)
    
    if cutoff == True:
        x = cutoff_intensity(x, -1000)
    
    if add_noise == True:
        ValueError('WRONG NOISE ADDITION CODE')
        x =  x + np.random.normal(0, sigma, x.shape) 

    if normalize == True:
        x = normalize_image(x)
    
    if expand_dim == True:
        x = np.expand_dims(x, axis = -1)
    # print('after adapt, shape of x is: ', x.shape)
    return x


def normalize_image(x, normalize_factor = 1000, image_max = 100, image_min = -100, final_max = 1, final_min = -1 , invert = False):
    # a common normalization method in CT
    # if you use (x-mu)/std, you need to preset the mu and std
    if invert == False:
        if isinstance(normalize_factor, int): # direct division
            return x.astype(np.float32) / normalize_factor
        else: # normalize_factor == 'equation'
            return (final_max - final_min) / (image_max - image_min) * (x.astype(np.float32) - image_min) + (final_min)
    else:
        if isinstance(normalize_factor, int): # direct division
            return x * normalize_factor
        else: # normalize_factor == 'equation'
            return (x - final_min) * (image_max - image_min) / (final_max - final_min) + image_min


def cutoff_intensity(x,cutoff_low = None, cutoff_high = None):
    xx = np.copy(x)

    if cutoff_low is not None and np.min(x) < cutoff_low:
        xx[x <= cutoff_low] = cutoff_low
    
    if cutoff_high is not None and np.max(x) > cutoff_high:
        xx[x >= cutoff_high] = cutoff_high
    return xx

# function: translate image
def translate_image(image, shift):
    assert len(shift) in [2, 3], "Shift must be a list of 2 elements for 2D or 3 elements for 3D"
    assert len(image.shape) in [2, 3], "Image must be either 2D or 3D"
    assert len(image.shape) == len(shift), "Shift dimensions must match image dimensions"

    fill_val = np.min(image)  # Fill value is the minimum value in the image
    translated_image = np.full_like(image, fill_val)  # Create an image filled with fill_val

    if image.ndim == 2:  # 2D image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                new_i = i - shift[0]
                new_j = j - shift[1]
                if 0 <= new_i < image.shape[0] and 0 <= new_j < image.shape[1]:
                    translated_image[new_i, new_j] = image[i, j]
    elif image.ndim == 3:  # 3D image
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    new_i = i - shift[0]
                    new_j = j - shift[1]
                    new_k = k - shift[2]
                    if 0 <= new_i < image.shape[0] and 0 <= new_j < image.shape[1] and 0 <= new_k < image.shape[2]:
                        translated_image[new_i, new_j, new_k] = image[i, j, k]
    else:
        raise ValueError("Image dimensions not supported")

    return translated_image


# function: rotate image
def rotate_image(image, degrees, order, fill_val = None):

    if fill_val is None:
        fill_val = np.min(image)
        
    if image.ndim == 2:  # 2D image
        assert isinstance(degrees, (int, float)), "Degrees should be a single number for 2D rotation"
        rotated_img = ndimage.rotate(image, degrees, reshape=False, mode='constant', cval=fill_val, order = order)

    elif image.ndim == 3:  # 3D image
        assert len(degrees) == 3 and all(isinstance(deg, (int, float)) for deg in degrees), "Degrees should be a list of three numbers for 3D rotation"
        # Rotate around x-axis
        rotated_img = ndimage.rotate(image, degrees[0], axes=(1, 2), reshape=False, mode='constant', cval=fill_val, order  = order)
        # Rotate around y-axis
        rotated_img = ndimage.rotate(rotated_img, degrees[1], axes=(0, 2), reshape=False, mode='constant', cval=fill_val, order = order)
        # Rotate around z-axis
        rotated_img = ndimage.rotate(rotated_img, degrees[2], axes=(0, 1), reshape=False, mode='constant', cval=fill_val, order = order)
    else:
        raise ValueError("Image must be either 2D or 3D")

    return rotated_img

    

def save_partial_volumes(img_list,file_name,slice_range = None): # only save some slices of an original CT volume
    for img_file in img_list:
        f = os.path.join(os.path.dirname(img_file),file_name)

        if os.path.isfile(f) == 1:
            print('already saved partial volume')
            continue
        
        x = nb.load(img_file)
        img = x.get_data()
        print(img_file,img.shape)
        

        if slice_range == None:
            # slice_range = [int(img.shape[-1]/2) - 30, int(img.shape[-1]/2) + 30]
            slice_range = [10,60]
        
        if img.shape[-1] < (slice_range[1] - slice_range[0]):
            print('THIS ONE DOES NOT HAVE ENOUGH SLICES, CONTINUE')
            continue
        
        img = img[:,:,slice_range[0]:slice_range[1]]

        # ff.make_folder([f])
        img = nb.Nifti1Image(img,x.affine)
        nb.save(img, f)


def downsample_crop_image(img_list, file_name, crop_size, factor = [2,2,1],):
    # crop_size = [128,128,z_dim]

    for img_file in img_list:
        f = os.path.join(os.path.dirname(img_file),file_name)
        print(img_file)

        if os.path.isfile(f) == 1:
            print('already saved partial volume')
            continue
        #
        x = nb.load(img_file)
        header = x.header
        spacing = x.header.get_zooms()
        affine = x.affine
        img = x.get_fdata()

        img_ds =  block_reduce(img, block_size = (factor[0] , factor[1], factor[2]), func=np.mean)
        img_ds = crop_or_pad(img_ds,crop_size, value = np.min(img_ds))

        # new parameters
        new_spacing = [spacing[0] * factor[0], spacing[1] * factor[1], spacing[2] * factor[2]]
        
        T = np.eye(4); T[0,0] = factor[0]; T [1,1] = factor[1]; T[2,2] = factor[2] 
        new_affine = np.dot(affine,T)
        new_header = header; new_header['pixdim'] = [-1, new_spacing[0], new_spacing[1], new_spacing[2],0,0,0,0]

        # save downsampled image
        recon_nb = nb.Nifti1Image(img_ds, new_affine, header  = new_header)
        nb.save(recon_nb, f)

    
def move_3Dimage(image, d):
    if len(d) == 3:  # 3D

        d0, d1, d2 = d
        S0, S1, S2 = image.shape

        start0, end0 = 0 - d0, S0 - d0
        start1, end1 = 0 - d1, S1 - d1
        start2, end2 = 0 - d2, S2 - d2

        start0_, end0_ = max(start0, 0), min(end0, S0)
        start1_, end1_ = max(start1, 0), min(end1, S1)
        start2_, end2_ = max(start2, 0), min(end2, S2)

        # Crop the image
        crop = image[start0_: end0_, start1_: end1_, start2_: end2_]
        crop = np.pad(crop,
                        ((start0_ - start0, end0 - end0_), (start1_ - start1, end1 - end1_),
                        (start2_ - start2, end2 - end2_)),
                        'constant')

    if len(d) == 2: # 2D
        d0, d1 = d
        S0, S1 = image.shape

        start0, end0 = 0 - d0, S0 - d0
        start1, end1 = 0 - d1, S1 - d1

        start0_, end0_ = max(start0, 0), min(end0, S0)
        start1_, end1_ = max(start1, 0), min(end1, S1)

        # Crop the image
        crop = image[start0_: end0_, start1_: end1_]
        crop = np.pad(crop,
                        ((start0_ - start0, end0 - end0_), (start1_ - start1, end1 - end1_)),
                        'constant')

    return crop


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
    
    
    
    