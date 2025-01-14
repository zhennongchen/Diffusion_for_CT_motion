# Inference Pipeline from dicom to corrected nii for motion correction

Dufan Wu

wudufan33@gmail.com

## Data
The pipeline receives thin slice (1.25mm) soft tissue CT images from Neurologica CT. The dicoms and models should be put in a folder like:

```
data_folder/
    dicoms/
        ID1/
            *.dcm
        ID2/
            *.dcm
        ...
    trained_models/
        model_20250110.pt   # model checkpoint
        bins.npy            # original HU bin value for reversible histogram equalization
        bins_mapped.npy     # target HU bin value for reversible histogram equalization
```

## Dockerfile
One should first run `/docker/docker_build.sh` to build the docker image

Then in `/docker/docker_run.sh`, change the following two lines:

```dockerfile
HOST_WORK_DIR="/home/local/PARTNERS/dw640/Diffusion_for_CT_motion"
HOST_DATA_DIR="/home/local/PARTNERS/dw640/mnt/CAMCA/home/dufan.wu/research_output/ct_motion_diffusion"
```

Change `HOST_WORK_DIR` to the directory of the overall code directory `Diffusion_for_CT_Motion` on the host. Change `HOST_DATA_DIR` to the directory of `data_folder` on the host in the previous section.

## Step 1. dcm2niix
Run `step_1_dcm2niix.py`. It will scan all the dicom folders under the input folder and convert them to nii.gz files in the nii folder. See the scripts for optional input parameters.

Input:
```
data_folder/
    dicoms/
        ID1/
            *.dcm
        ID2/
            *.dcm
        ...
```

Output:
```
data_folder/
    niis/
        ID1.nii.gz
        ID2.nii.gz
        ...
```

## Step 2. Resampling
Run `step_2_resampling.py`. It will resample the input images to a pixel size of (1mm, 1mm, 2.5mm). It uses nearest neighbor interpolation in the x and y direction, and slice averaging in the z direction. Because the input is 1.25mm slice thickness, so it averages every two slices in z.

See the scripts for optional parameters.

Input:
```
data_folder/
    niis/
        ID1.nii.gz
        ID2.nii.gz
        ...
```

Output:
```
data_folder/
    preprocessed/
        ID1.nii.gz
        ID2.nii.gz
        ...
```

## Step 3. Inference
Run `step_3_inference.py`. It will do the network diffusion model inference. It breaks the volume into patches in the z direction, do the inference for each patch, and then stitch them together. 

The key parameters are

- `--device`: which GPU to run the model;
- `--remove_intermediate`: If the intermediate results (per slab output) will be removed after inference
- `--num_sample_steps`: number of the reverse diffusion timesteps
- `--img_size_3d`: the patch size in (x, y, z)
- `--slice_step_size`: the z step size of the patches.

By default, patch has nz=20 with a stepsize of 10. During the stitch, the central 10 slices of each patch (except for the first and last patches) will be preserved. No overlap will happen during the stitch. This can be further explored for acceleration, e.g., using a larger stepsize.

For a patch size of 256x256x20 with 50 time steps, tt costs around 19GB of GPU memory on a V100-32GB or A100-40GB. It may be fit into a 16GB GPU but the speed may be slower. The time is around 75 seconds per patch on A100 and 150 seconds per patch on V100. 

Input:
```
data_folder/
    preprocessed/
        ID1.nii.gz
        ID2.nii.gz
        ...
```

Output:
```
data_folder/
    output/
        ID1.nii.gz
        ID2.nii.gz
        ID1/                # intermediate outputs, will be removed after each image if remove_intermediate set to 1
            *.nii.gz
        ID2/                # intermediate outputs, will be removed after each image if remove_intermediate set to 1
            *.nii.gz
        ...
```
