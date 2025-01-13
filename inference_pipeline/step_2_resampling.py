'''
Resampling to 1mmx1mm in the axial and averaing to 2.5mm in the z
'''

# %%
import os
import argparse
import subprocess
import sys

import numpy as np
import nibabel as nb

import Diffusion_for_CT_motion.utils.Data_processing as Data_processing

from locations import default_nii_dir, default_preprocessed_dir


# %%
def get_args(default_args=[]):
    parser = argparse.ArgumentParser(description='Resample the nifti files')
    parser.add_argument('--nii_dir', default=default_nii_dir,
                        help='The directory where the nifti files are stored')
    parser.add_argument('--preprocessed_dir', default=default_preprocessed_dir,
                        help='The directory where the resampled nifti files will be stored')
    parser.add_argument('--in_plane_res_mm', type=float, default=1,
                        help='The resolution in the x direction')
    parser.add_argument('--res_z_mm', type=float, default=2.5,
                        help='The resolution in the z direction')
    parser.add_argument('--slice_factor', type=int, default=2)

    if 'ipykernel' in sys.argv[0]:
        args = parser.parse_args(default_args)
    else:
        args = parser.parse_args()

    args.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    args.datetime = subprocess.check_output(['date', '+%Y-%m-%d %H:%M:%S']).decode('utf-8').strip()
    args.user = subprocess.check_output(['whoami']).decode('utf-8').strip()
    args.sys_argv = sys.argv

    for k in vars(args):
        print(k, '=', getattr(args, k), flush=True)

    return args


# %%
def main(args):
    nii_dir = args.nii_dir
    output_dir = args.preprocessed_dir
    os.makedirs(output_dir, exist_ok=True)

    in_plane_res = args.in_plane_res_mm
    res_z = args.res_z_mm
    slice_factor = args.slice_factor

    filenames = os.listdir(nii_dir)
    filenames = [f for f in filenames if f.endswith('.nii.gz')]

    print('Resampling', len(filenames), 'files', flush=True)

    for k, filename in enumerate(filenames):
        print(k, filename, flush=True)

        img = nb.load(os.path.join(nii_dir, filename))

        print('Original shape:', img.get_fdata().shape)
        print('Original pixel dim:', img.header.get_zooms()[:3])

        # xy resamping with nearest interpolation
        hr_resample = Data_processing.resample_nifti(
            img,
            order=3,
            mode='nearest',
            cval=np.min(img.get_fdata()),
            in_plane_resolution_mm=in_plane_res,
            slice_thickness_mm=res_z / slice_factor
        )
        hr_resample = nb.Nifti1Image(hr_resample.get_fdata(), affine=hr_resample.affine, header=hr_resample.header)

        # z resample using averaging
        img_data = hr_resample.get_fdata()
        # averaging every two slices in z
        resampled_data = img_data[..., :img_data.shape[-1] // slice_factor * slice_factor]
        resampled_data = np.mean(resampled_data.reshape(resampled_data.shape[:2] + (-1, slice_factor)), axis=-1)

        # setup new header and image
        new_affine = hr_resample.affine.copy()
        new_affine[2, 2] *= slice_factor
        new_header = hr_resample.header.copy()
        new_header.set_zooms([1, 1, res_z])
        resampled_img = nb.Nifti1Image(resampled_data, new_affine, new_header)

        print('Final shape', resampled_img.get_fdata().shape)
        print('Final pixel dim:', resampled_img.header.get_zooms()[:3])

        nb.save(resampled_img, os.path.join(output_dir, filename))

    print('Done', flush=True)


# %%
if __name__ == '__main__':
    args = get_args([])
    res = main(args)
