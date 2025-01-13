'''
Convert the dicoms to nii
'''

# %%
import os
import argparse
import subprocess
import sys

import shutil

from locations import default_dicom_dir, default_nii_dir


# %%
def get_args(default_args=[]):
    parser = argparse.ArgumentParser(description='Convert dicoms to nii')
    parser.add_argument('--dicom_dir', default=default_dicom_dir,
                        help='Directory containing dicoms folders, each folder has single series')
    parser.add_argument('--nii_dir', default=default_nii_dir,
                        help='Directory to save nii files')

    if 'ipykernel' in sys.argv[0]:
        args = parser.parse_args(default_args)
    else:
        args = parser.parse_args()

    args.git_hash = subprocess.check_output(["git", "describe", "--always"]).strip().decode('utf-8')
    args.datetime = subprocess.check_output(["date", "+%Y-%m-%d %H:%M:%S"]).strip().decode('utf-8')
    args.user = subprocess.check_output(["whoami"]).strip().decode('utf-8')
    args.sys_argv = sys.argv

    for k in vars(args):
        print(k, '=', getattr(args, k), flush=True)

    return args


# %%
def main(args):
    dicom_dir = args.dicom_dir
    nii_dir = args.nii_dir
    os.makedirs(nii_dir, exist_ok=True)

    folders = os.listdir(dicom_dir)
    folders = [f for f in folders if os.path.isdir(os.path.join(dicom_dir, f))]

    print('Converting dicoms to nii')
    for k, folder in enumerate(folders):
        print('Converting', folder, k + 1, 'of', len(folders))

        input_dir = os.path.join(dicom_dir, folder)
        tmp_dir = os.path.join(nii_dir, folder + '_tmp')
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            subprocess.run(['dcm2niix', '-z', 'y', '-m', 'y', '-o', tmp_dir, '-f', folder, input_dir])
        except Exception as e:
            print('Error converting', folder, e)

        shutil.copyfile(os.path.join(tmp_dir, folder + '.nii.gz'), os.path.join(nii_dir, folder + '.nii.gz'))
        shutil.rmtree(tmp_dir)
    print('Done')


# %%
if __name__ == '__main__':
    args = get_args()
    res = main(args)
