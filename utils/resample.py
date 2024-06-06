import Diffusion_for_CT_motion.utils.Data_processing as Data_processing
import Diffusion_for_CT_motion.utils.functions_collection as ff
from Diffusion_for_CT_motion.Build_lists import Build_list


import argparse
import os
import numpy as np
import nibabel as nb
import shutil
import pandas as pd
from skimage.measure import block_reduce
import scipy.ndimage as ndimage
from scipy.ndimage import zoom

main_folder = '/mnt/camca_NAS/Portable_CT_data'


# new way of resampling (Averaging)
############ resample data for simulation
patient_list = ff.find_all_target_files(['*/*'], os.path.join(main_folder, 'simulations_202404/simulated_all_motion_v1'))

### slice_portable is the slice number before resampling
for i in range(0,patient_list.shape[0]):
    patient = patient_list[i]
    patient_subid = os.path.basename(patient)
    patient_id = os.path.basename(os.path.dirname(patient))
    folders = ff.find_all_target_files(['random_*', 'static'], patient)
    
    for folder in folders:
        motion_name = os.path.basename(folder)
        print(patient_id, patient_subid, motion_name)

        save_folder = os.path.join(folder, 'image_data')

        if os.path.isfile(os.path.join(save_folder, 'recon_resample_avg.nii.gz')):
            print('did this')
            continue

        img = nb.load(os.path.join(save_folder, 'recon.nii.gz'))
        img_data = img.get_fdata()
        print('original shape:', img_data.shape)
        # original pixel dim
        pixel_dim = img.header.get_zooms()[:3]
        print('original pixel dim:', pixel_dim)
        # assert it's [1,1,1]
        assert pixel_dim[0] == 1 and pixel_dim[1] == 1 and pixel_dim[2] == 1

        # do for x and y resampling:
        new_dim = [1,1,1.25]

        hr_resample = Data_processing.resample_nifti(img, order=3,  mode = 'nearest',  cval = np.min(img.get_fdata()), in_plane_resolution_mm=new_dim[0], slice_thickness_mm=new_dim[-1])
        hr_resample = nb.Nifti1Image(hr_resample.get_fdata(), affine=hr_resample.affine, header=hr_resample.header)

        print('last shape:', hr_resample.get_fdata().shape)
        print('new pixel dim:', hr_resample.header.get_zooms()[:3])

        # do for z resampling (using averaging)
        img_data = hr_resample.get_fdata()
        affine = hr_resample.affine
        pixel_dim = hr_resample.header.get_zooms()[:3]
        header = hr_resample.header

        new_z_res = 2.5

        slice_factor = new_z_res // pixel_dim[-1]

        # Calculate the new shape
        new_shape = list(img_data.shape)
        new_shape[2] = int(img_data.shape[2] // slice_factor)
        print('new shape:', new_shape)  
        # Initialize the resampled data
        resampled_data = np.zeros(new_shape)

        for i in range(new_shape[2]):
            start_slice = int(i * slice_factor)
            end_slice = int((i + 1) * slice_factor)

            resampled_data[:, :, i] = np.mean(img_data[:, :, start_slice:end_slice], axis=2)

        # Update the affine and header for the new voxel size
        new_affine = affine.copy()
        new_affine[2, 2] = affine[2, 2] * slice_factor
        new_header = header.copy()
        new_header.set_zooms([1,1, pixel_dim[-1] * slice_factor])
    
        # Create and save the new NIfTI image
        resampled_img = nb.Nifti1Image(resampled_data, new_affine, new_header)
        # print new pixel dim:
        print('final pixel dim:', resampled_img.header.get_zooms()[:3])
        assert resampled_img.header.get_zooms()[0] == 1 and resampled_img.header.get_zooms()[1] == 1 and resampled_img.header.get_zooms()[2] == 2.5
        assert resampled_img.get_fdata().shape[2] >= 60
        
        nb.save(resampled_img, os.path.join(save_folder, 'recon_resample_avg.nii.gz'))



############ resample data for portable & fixed CT (Testing data)
# data_folder = os.path.join(main_folder, 'nii_imgs_202404', 'motion')
# patient_sheet = pd.read_excel(os.path.join(main_folder,'Patient_list', 'NEW_CT_concise_collected_portable_motion_w_fixed_CT_info_edited.xlsx'),dtype={'Patient_ID': str, 'Patient_subID': str, 'Patient_ID_fixed': str, 'Patient_subID_fixed': str})
# patient_sheet['use_portable'] = patient_sheet['use_portable'].fillna(0)
# patient_sheet = patient_sheet[(patient_sheet['use_portable'] != 0) & (patient_sheet['use_portable'] != 'no')]
# print(patient_sheet.shape)

# portable = True

# ### slice_portable is the slice number before resampling
# for i in range(0,patient_sheet.shape[0]):
#     row = patient_sheet.iloc[i]
   
#     if portable:
#         patient_id = row['Patient_ID']
#         patient_subid = row['Patient_subID']
#         slice_num = eval(row['slice_portable'])
#         filename = 'img.nii.gz'
#     else:
#         patient_id = row['Patient_ID_fixed']
#         patient_subid = row['Patient_subID_fixed']
#         slice_num = eval(row['slice_fixed'])
#         filename = row['use_fixed']+'.nii.gz'
    
#     print(patient_id, patient_subid,  slice_num)

#     if os.path.isfile(os.path.join(data_folder, patient_id, patient_subid, 'portable/img_resample_avg.nii.gz')) and portable:
#         print('did this')
#         continue
#     if os.path.isfile(os.path.join(data_folder, patient_id, patient_subid, 'fixed/img_resample_avg.nii.gz')) and not portable:
#         print('did this')
#         continue

#     if portable:
#         img = nb.load(os.path.join(data_folder, patient_id, patient_subid, 'portable', filename))
#     else:
#         img = nb.load(os.path.join(data_folder, patient_id, patient_subid, 'fixed', filename))

#     img_data = img.get_fdata()
#     print('original shape:', img_data.shape)
#     # if dim is 4:
#     if len(img_data.shape) == 4:
#         print('Dimension is 4')
#         img_data = img_data[:,:,:,0]

#     img_data = img_data[:,:,slice_num[0]:slice_num[1]]
#     print('after slice:', img_data.shape)
#     img = nb.Nifti1Image(img_data, affine=img.affine, header=img.header)
#     # original pixel dim
#     pixel_dim = img.header.get_zooms()[:3]
#     print('original pixel dim:', pixel_dim)

#     # do for x and y resampling:
#     new_dim = [1,1,pixel_dim[-1]]

#     hr_resample = Data_processing.resample_nifti(img, order=3,  mode = 'nearest',  cval = np.min(img.get_fdata()), in_plane_resolution_mm=new_dim[0], slice_thickness_mm=new_dim[-1])
#     hr_resample = nb.Nifti1Image(hr_resample.get_fdata(), affine=hr_resample.affine, header=hr_resample.header)

#     print('last shape:', hr_resample.get_fdata().shape)
#     print('new pixel dim:', hr_resample.header.get_zooms()[:3])

#     # do for z resampling (using averaging)
#     img_data = hr_resample.get_fdata()
#     affine = hr_resample.affine
#     pixel_dim = hr_resample.header.get_zooms()[:3]
#     header = hr_resample.header

#     new_z_res = 2.5

#     slice_factor = new_z_res // pixel_dim[-1]

#     # Calculate the new shape
#     new_shape = list(img_data.shape)
#     new_shape[2] = int(img_data.shape[2] // slice_factor)
#     print('new shape:', new_shape)  
#     # Initialize the resampled data
#     resampled_data = np.zeros(new_shape)

#     for i in range(new_shape[2]):
#         start_slice = int(i * slice_factor)
#         end_slice = int((i + 1) * slice_factor)

#         resampled_data[:, :, i] = np.mean(img_data[:, :, start_slice:end_slice], axis=2)

#     # Update the affine and header for the new voxel size
#     new_affine = affine.copy()
#     new_affine[2, 2] = affine[2, 2] * slice_factor
#     new_header = header.copy()
#     new_header.set_zooms([1,1, pixel_dim[-1] * slice_factor])
    
#     # Create and save the new NIfTI image
#     resampled_img = nb.Nifti1Image(resampled_data, new_affine, new_header)
#     # print new pixel dim:
#     print('final pixel dim:', resampled_img.header.get_zooms()[:3])

#     if portable:
#         assert resampled_img.get_fdata().shape[2] == 50 or resampled_img.get_fdata().shape[2] == 55 or resampled_img.get_fdata().shape[2] == 60
    
#     if portable:
#         nb.save(resampled_img, os.path.join(data_folder, patient_id, patient_subid, 'portable/img_resample_avg.nii.gz'))
#     else:
#         nb.save(resampled_img, os.path.join(data_folder, patient_id, patient_subid, 'fixed/img_resample_avg.nii.gz'))


         