import numpy as np
import nibabel as nb
import os
from skimage.measure import block_reduce
import Diffusion_for_CT_motion.simulation_data_generation.functions as ff


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


def normalize_image(x):
    # a common normalization method in CT
    # if you use (x-mu)/std, you need to preset the mu and std
    
    return x.astype(np.float32) / 1000


def cutoff_intensity(x,cutoff):
    if np.min(x) < cutoff:
        x[x<cutoff] = cutoff
    return x


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