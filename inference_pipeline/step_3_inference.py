'''
Inference using trained model
'''

# %%
import argparse
import os
import sys
import subprocess

import shutil
import nibabel as nb
import numpy as np

import Diffusion_for_CT_motion.diffusion_models.conditional_diffusion_3D as ddpm_3D
import Diffusion_for_CT_motion.diffusion_models.conditional_EDM_3D as edm
import Diffusion_for_CT_motion.utils.Generator as Generator

from locations import default_preprocessed_dir, default_output_dir, default_model_filename
from locations import default_hist_eq_bins, default_hist_eq_mapped


# %%
def get_args(default_args=[]):
    parser = argparse.ArgumentParser(description='Inference using trained model')
    parser.add_argument('--input_dir', default=default_preprocessed_dir)
    parser.add_argument('--output_dir', default=default_output_dir)
    parser.add_argument('--checkpoint', default=default_model_filename)
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--remove_intermediate', type=int, default=1)

    parser.add_argument('--num_sample_steps', type=int, default=50,
                        help='Number of sampling steps for the diffusion model')

    parser.add_argument('--img_size_3d', type=int, nargs=3, default=[256, 256, 20],
                        help='Patch size to be processed in 3D [nx, ny, nz]')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--slice_step_size', type=int, default=10,
                        help='Step size for slicing the 3D volume')

    # normalization
    parser.add_argument('--histogram_equalization', type=int, default=1)
    parser.add_argument('--histogram_equalization_bins', default=default_hist_eq_bins)
    parser.add_argument('--histogram_equalization_mapped', default=default_hist_eq_mapped)
    parser.add_argument('--background_cutoff', type=int, default=-1000)
    parser.add_argument('--maximum_cutoff', type=int, default=2000)
    parser.add_argument('--normalize_factor', default='equation')
    parser.add_argument('--clip_range', type=float, nargs=2, default=[-1, 1],
                        help='Grayscale clip after normalization')

    if 'ipykernel' in sys.argv[0]:
        args = parser.parse_args(default_args)
    else:
        args = parser.parse_args()

    args.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    args.datetime = subprocess.check_output(['date', '+%Y-%m-%d_%H-%M-%S']).strip().decode('utf-8')
    args.user = subprocess.check_output(['whoami']).strip().decode('utf-8')
    args.sys_argv = sys.argv

    for k in vars(args):
        print(k, '=', getattr(args, k), flush=True)

    return args


# %%
def get_model(img_size_3d, clip_range, num_sample_steps):
    model = ddpm_3D.Unet3D(
        init_dim=64,
        channels=1,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False,
        conditional_diffusion=True,
        full_attn=(None, None, False, True),
    )

    diffusion_model = edm.EDM(
        model,
        image_size=img_size_3d,
        num_sample_steps=num_sample_steps,
        clip_or_not=True,
        clip_range=clip_range,
    )

    return diffusion_model


# %%
def get_slice_range_list(filename, img_size_3d, slice_step_size):
    # load the image and check the number of slices
    img = nb.load(filename)
    data = img.get_fdata()
    n_slices = data.shape[-1]
    nz_patch = img_size_3d[-1]

    start_slices = np.arange(0, n_slices, slice_step_size)
    start_slices[start_slices + nz_patch > n_slices] = n_slices - nz_patch
    start_slices = np.unique(start_slices)

    slice_range_list = [[start, start + nz_patch] for start in start_slices]

    return slice_range_list


# %%
def combine_slices(output_filename, original_filename, slice_dir, slice_range_list):
    # stitch the slices using the central half slices from each slab.
    res = []
    islice = 0  # current starting slice in the result volume
    for i, slice_range in enumerate(slice_range_list):
        slabname = os.path.join(slice_dir, 'pred_{}_{}.nii.gz'.format(slice_range[0], slice_range[1]))
        slab = nb.load(slabname).get_fdata()

        # the starting slice in the loaded slab
        istart = islice - slice_range[0]
        if i == len(slice_range_list) - 1:
            # load all slices
            iend = slab.shape[-1]
        else:
            # the ending slab will be the average between the end of the current and the start of the next
            iend = (slice_range_list[i][1] + slice_range_list[i + 1][0]) // 2 - slice_range[0]

        res.append(slab[..., istart:iend])
        islice += iend - istart

    res = np.concatenate(res, axis=-1)

    # get reference information from the original image
    ori_img = nb.load(original_filename)
    res_img = nb.Nifti1Image(res, ori_img.affine, ori_img.header)

    # save the result
    nb.save(res_img, output_filename)


# %%
def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print('Define model...', flush=True)
    diffusion_model = get_model(args.img_size_3d, args.clip_range, args.num_sample_steps)

    filenames = os.listdir(input_dir)
    filenames = [os.path.join(input_dir, f) for f in filenames if f.endswith('.nii.gz')]
    print('Processing', len(filenames), 'files...', flush=True)

    for i, filename in enumerate(filenames):
        print('Processing', i, os.path.basename(filename), flush=True)

        slice_range_list = get_slice_range_list(filename, args.img_size_3d, args.slice_step_size)
        print('There are {} slabs to process'.format(len(slice_range_list)), flush=True)
        print(slice_range_list, flush=True)

        # tmp output directory
        tmp_output_dir = os.path.join(output_dir, os.path.basename(filename)[:-7])
        os.makedirs(tmp_output_dir, exist_ok=True)

        for k, slice_range in enumerate(slice_range_list):
            print('Processing slice range', k, slice_range, flush=True)

            generator = Generator.Dataset_dual_patch(
                np.array([filename]),
                np.array([filename]),
                image_size_3D=args.img_size_3d,
                patch_size=args.patch_size,
                patch_stride=1,
                original_patch_num=1,
                random_sampled_patch_num=0,
                patch_selection=None,
                slice_number=slice_range[1] - slice_range[0],
                slice_start=int(slice_range[0]),

                histogram_equalization=args.histogram_equalization,
                background_cutoff=args.background_cutoff,
                maximum_cutoff=args.maximum_cutoff,
                normalize_factor=args.normalize_factor,

                histogram_bins=args.histogram_equalization_bins,
                histogram_bins_mapped=args.histogram_equalization_mapped,
            )

            # sample:
            sampler = edm.Sampler(
                diffusion_model,
                generator,
                image_size=args.img_size_3d,
                batch_size=1,
                histogram_bins=args.histogram_equalization_bins,
                histogram_bins_mapped=args.histogram_equalization_mapped,
            )

            sampler.sample_3D_w_trained_model(
                trained_model_filename=args.checkpoint,
                ground_truth_image_file=filename,
                motion_image_file=filename,
                save_file=os.path.join(tmp_output_dir, 'pred_{}_{}.nii.gz'.format(slice_range[0], slice_range[1])),
                slice_range=slice_range,
                save_gt_motion=False,
                not_start_from_first_slice=False
            )

        # combine the slices
        print('Combining slices...', flush=True)
        combine_slices(
            os.path.join(output_dir, os.path.basename(filename)),
            filename,
            tmp_output_dir,
            slice_range_list
        )

        if args.remove_intermediate:
            print('Removing intermediate files...', flush=True)
            shutil.rmtree(tmp_output_dir)

    print('Done.', flush=True)


# %%
if __name__ == '__main__':
    args = get_args(['--remove_intermediate', '0'])
    res = main(args)
