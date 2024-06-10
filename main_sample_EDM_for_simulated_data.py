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
epoch = 82
trained_model_filename = '/mnt/camca_NAS/diffusion_ct_motion/models/portable_EDM_patch_3Dmotion_hist_v1/models/model-' + str(epoch) + '.pt'
save_folder = os.path.join('/mnt/camca_NAS/diffusion_ct_motion/models', trial_name, 'pred_images_portable_simulated2'); os.makedirs(save_folder, exist_ok=True)

########################### set the data path!
data_sheet = os.path.join('/mnt/camca_NAS/diffusion_ct_motion/data/Patient_list/Patient_list_train_test_simulated_all_motion_v1.xlsx')
b = Build_list.Build(data_sheet)
_,_,_,_, _,_, x0_list, _, condition_list, _, _,_,_ = b.__build__(batch_list = [4])  # batch 4 is the testing batch
n = ff.get_X_numbers_in_interval(total_number = x0_list.shape[0],start_number = 0,end_number = 20, interval = 20) # each case has 20 motion samples
x0_list = x0_list[n]; condition_list = condition_list[n]
simulated_data = True

# default parameters, don't change unless necessary
image_size_3D = [256,256,20]
patch_size = 256 # apply on whole image
slice_range_list = [[0,20],[10,30],[20,40],[30,50]] # do for every 20-slice stack, can change to 25 or some other numbers

histogram_equalization = True
background_cutoff = -1000
maximum_cutoff = 2000
normalize_factor = 'equation'
clip_range = [-1,1]

save_gt_motion = True
simulated_data = True
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
    num_sample_steps =100,
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
    affine = nb.load(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '_slice30to50.nii.gz')).affine
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

    # gt
    slice_0_20_image = nb.load(os.path.join(save_folder_case, 'gt_slice0to20.nii.gz')).get_fdata()
    slice_10_30_image = nb.load(os.path.join(save_folder_case, 'gt_slice10to30.nii.gz')).get_fdata()
    slice_20_40_image = nb.load(os.path.join(save_folder_case, 'gt_slice20to40.nii.gz')).get_fdata()
    slice_30_50_image = nb.load(os.path.join(save_folder_case, 'gt_slice30to50.nii.gz')).get_fdata()
        
    final_image = np.zeros((slice_0_20_image.shape[0], slice_0_20_image.shape[1], 50))
    final_image[:,:,0:20] = slice_0_20_image[:,:,0:20]
    final_image[:,:,20:30] = slice_10_30_image[:,:,10:20]
    final_image[:,:,30:40] = slice_20_40_image[:,:,10:20]
    final_image[:,:,40:50] = slice_30_50_image[:,:,10:20]
    nb.save(nb.Nifti1Image(final_image,affine ), os.path.join(save_folder_case, 'gt.nii.gz'))

    # motion
    slice_0_20_image = nb.load(os.path.join(save_folder_case, 'motion_slice0to20.nii.gz')).get_fdata()
    slice_10_30_image = nb.load(os.path.join(save_folder_case, 'motion_slice10to30.nii.gz')).get_fdata()
    slice_20_40_image = nb.load(os.path.join(save_folder_case, 'motion_slice20to40.nii.gz')).get_fdata()
    slice_30_50_image = nb.load(os.path.join(save_folder_case, 'motion_slice30to50.nii.gz')).get_fdata()

    final_image = np.zeros((slice_0_20_image.shape[0], slice_0_20_image.shape[1], 50))
    final_image[:,:,0:20] = slice_0_20_image[:,:,0:20]
    final_image[:,:,20:30] = slice_10_30_image[:,:,10:20]
    final_image[:,:,30:40] = slice_20_40_image[:,:,10:20]
    final_image[:,:,40:50] = slice_30_50_image[:,:,10:20]
    nb.save(nb.Nifti1Image(final_image,affine ), os.path.join(save_folder_case, 'motion.nii.gz'))


