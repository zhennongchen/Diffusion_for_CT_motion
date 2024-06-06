import sys
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np 
import nibabel as nb
import Diffusion_for_CT_motion.diffusion_models.conditional_DDPM_3D as ddpm_3D
import Diffusion_for_CT_motion.diffusion_models.conditional_EDM_3D as edm
import Diffusion_for_CT_motion.utils.functions_collection as ff
import Diffusion_for_CT_motion.Build_lists.Build_list as Build_list
import Diffusion_for_CT_motion.utils.Generator as Generator


########################### set the trial name and trained model path
trial_name = 'portable_EDM_patch_3Dmotion_hist_v1'
epoch = 49
trained_model_filename = '/mnt/camca_NAS/diffusion_ct_motion/models/portable_EDM_patch_3Dmotion_hist_v1/models/model-' + str(epoch) + '.pt'
save_folder = os.path.join('/mnt/camca_NAS/diffusion_ct_motion/models', trial_name, 'pred_images_portable_real_resample_avg'); os.makedirs(save_folder, exist_ok=True)

########################### set the data path!
data_sheet = os.path.join('/mnt/camca_NAS/diffusion_ct_motion/data/Patient_list/Patient_list_real_portable_CT_202404_resample_avg.xlsx')
b = Build_list.Build(data_sheet)
_,_,_,_, _,_, x0_list, _, condition_list, _, _,_,_ = b.__build__(batch_list = [5])  # x0 is motino-free, condition is motion-corrupted
print(x0_list.shape)
simulated_data = False

# default parameters, don't change unless necessary
image_size_3D = [256,256,20]
patch_size = 256 # apply on whole image
slice_range_list_for_50 = [[0,20],[10,30],[20,40],[30,50]]  # in real data the slice number is 50, 55 or 60
slice_range_list_for_60 = [[0,20],[10,30],[20,40],[30,50],[40,60]]
slice_range_list_for_55 = [[0,20],[10,30],[20,40],[30,50],[35,55]]

histogram_equalization = True
background_cutoff = -1000
maximum_cutoff = 2000
normalize_factor = 'equation'
clip_range = [-1,1]

save_gt_motion = True
###########

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
    image_size = image_size_3D,
    num_sample_steps = 100,
    clip_or_not = True,
    clip_range = clip_range,
)

for i in range(0,x0_list.shape[0]):
    
    x0_file = x0_list[i]
    condition_file = condition_list[i]

    motion_name = os.path.basename(os.path.dirname(os.path.dirname(condition_file)))
    patient_subid = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(condition_file))))
    patient_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(condition_file)))))

    print(i,patient_id, patient_subid, motion_name)

    ff.make_folder([os.path.join(save_folder, patient_id), os.path.join(save_folder, patient_id, patient_subid), os.path.join(save_folder, patient_id, patient_subid, motion_name)])

    save_folder_case = os.path.join(save_folder, patient_id, patient_subid, motion_name, 'epoch' + str(epoch)); os.makedirs(save_folder_case, exist_ok=True)

    # check the slice number:
    img = nb.load(condition_file).get_fdata()
    slice_num = img.shape[2]
    if slice_num == 50:
        slice_range_list = slice_range_list_for_50
    elif slice_num == 60:
        slice_range_list = slice_range_list_for_60
    elif slice_num == 55:
        slice_range_list = slice_range_list_for_55
    print('slice_num:', slice_num, 'slice_range_list:', slice_range_list)

    if os.path.isfile(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '.nii.gz')) == 0:
    
        for slice_range in slice_range_list:
            
            generator = Generator.Dataset_dual_patch(
                np.array([x0_file]),
                np.array([condition_file]),
                image_size_3D = image_size_3D,
                patch_size = patch_size,
                patch_stride = 1,
                original_patch_num = 1,
                random_sampled_patch_num = 0,
                patch_selection = None,
                slice_number = slice_range[1] - slice_range[0],
                slice_start = (slice_range[0] + [0 if simulated_data == False else 10][0]),

                histogram_equalization = histogram_equalization,
                background_cutoff = background_cutoff, 
                maximum_cutoff = maximum_cutoff,
                normalize_factor = normalize_factor,)

            
            # sample:
            sampler = edm.Sampler(
                diffusion_model,
                generator,
                image_size = image_size_3D,
                batch_size = 1)

            sampler.sample_3D_w_trained_model(trained_model_filename=trained_model_filename, 
                                        ground_truth_image_file= x0_file,
                                        motion_image_file =  condition_file, 
                                        save_file = os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '_slice' + str(slice_range[0]) +'to' + str(slice_range[1])+ '.nii.gz'),
                                        slice_range = slice_range, 
                                        save_gt_motion = save_gt_motion,
                                        not_start_from_first_slice = simulated_data)
  
    print('already done')


    # some post-processing - dummy code to put stacks together
    affine = nb.load(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '_slice30to50.nii.gz')).affine
    if slice_num == 50:
        slice_0_20_image = nb.load(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '_slice0to20.nii.gz')).get_fdata()
        slice_10_30_image = nb.load(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '_slice10to30.nii.gz')).get_fdata()
        slice_20_40_image = nb.load(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '_slice20to40.nii.gz')).get_fdata()
        slice_30_50_image = nb.load(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '_slice30to50.nii.gz')).get_fdata()

        final_image = np.zeros((slice_0_20_image.shape[0], slice_0_20_image.shape[1], 50))
        final_image[:,:,0:15] = slice_0_20_image[:,:,0:15]
        final_image[:,:,15:25] = slice_10_30_image[:,:,5:15]
        final_image[:,:,25:35] = slice_20_40_image[:,:,5:15]
        final_image[:,:,35:50] = slice_30_50_image[:,:,5:20]
        nb.save(nb.Nifti1Image(final_image,affine ), os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '.nii.gz'))
    elif slice_num == 60:
        slice_0_20_image = nb.load(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '_slice0to20.nii.gz')).get_fdata()
        slice_10_30_image = nb.load(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '_slice10to30.nii.gz')).get_fdata()
        slice_20_40_image = nb.load(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '_slice20to40.nii.gz')).get_fdata()
        slice_30_50_image = nb.load(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '_slice30to50.nii.gz')).get_fdata()
        slice_40_60_image = nb.load(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '_slice40to60.nii.gz')).get_fdata()

        final_image = np.zeros((slice_0_20_image.shape[0], slice_0_20_image.shape[1], 60))
        final_image[:,:,0:15] = slice_0_20_image[:,:,0:15]
        final_image[:,:,15:25] = slice_10_30_image[:,:,5:15]
        final_image[:,:,25:35] = slice_20_40_image[:,:,5:15]
        final_image[:,:,35:45] = slice_30_50_image[:,:,5:15]
        final_image[:,:,45:60] = slice_40_60_image[:,:,5:20]
        nb.save(nb.Nifti1Image(final_image,affine ), os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '.nii.gz'))
    elif slice_num == 55:
        slice_0_20_image = nb.load(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '_slice0to20.nii.gz')).get_fdata()
        slice_10_30_image = nb.load(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '_slice10to30.nii.gz')).get_fdata()
        slice_20_40_image = nb.load(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '_slice20to40.nii.gz')).get_fdata()
        slice_30_50_image = nb.load(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '_slice30to50.nii.gz')).get_fdata()
        slice_35_55_image = nb.load(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '_slice35to55.nii.gz')).get_fdata()

        final_image = np.zeros((slice_0_20_image.shape[0], slice_0_20_image.shape[1], 55))
        final_image[:,:,0:15] = slice_0_20_image[:,:,0:15]
        final_image[:,:,15:25] = slice_10_30_image[:,:,5:15]
        final_image[:,:,25:35] = slice_20_40_image[:,:,5:15]
        final_image[:,:,35:45] = slice_30_50_image[:,:,5:15]
        final_image[:,:,45:55] = slice_35_55_image[:,:,10:20]
        nb.save(nb.Nifti1Image(final_image,affine ), os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '.nii.gz'))


    # motion
    motion_image = nb.load(condition_file).get_fdata()
    nb.save(nb.Nifti1Image(motion_image,affine ), os.path.join(save_folder_case, 'motion.nii.gz'))

