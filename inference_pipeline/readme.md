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