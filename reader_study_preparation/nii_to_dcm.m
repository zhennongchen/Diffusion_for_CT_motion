%%% this script is for those patients with multiple folders for cine SAX
%%% data in the raw data
clear all; close all; clc;
addpath(genpath('/Users/zhennongchen/Documents/GitHub/CMR_HFpEF_Analysis/matlab'));
addpath(genpath('Users/zhennongchen/Documents/GitHub/Volume_Rendering_by_DL/matlab'));
data_path1 = '/Volumes/Camca/home/ZC/diffusion_ct_motion/examples/real_data';
data_path2 = '/Volumes/Camca/home/ZC/Portable_CT_data/IRB2022P002233_collected_202404';

patient_list = [data_path1, '/list/NEW_CT_concise_collected_portable_motion_w_fixed_CT_info_Diffusion_results_blinded_reference.xlsx'];
patient_list = readtable(patient_list);

% ID_list = [];
% timeframe_nums = [];
% dimension_error = [];
% [folders] = Find_all_folders(data_path);
%%
for i = 1:size(patient_list,1)
patient_id = patient_list{i,"Patient_ID"}{1};
patient_subid = patient_list{i,"Patient_subID"}{1};
disp(patient_id)
disp(patient_subid)
dicom_image_folder = [data_path2, '/', patient_id, '/', patient_subid];

% load the dicom image for metadata
dicom_image_subfolder = Find_all_folders(dicom_image_folder);
dicom_image_subfolder = [dicom_image_folder, '/', dicom_image_subfolder.name];
dicom_image_files = Find_all_files(dicom_image_subfolder);
filename = [dicom_image_subfolder, '/', dicom_image_files(1,:).name];
metadata = dicominfo(filename);

% read the nii files
nii_folder = [data_path1, '/data/', patient_id, '/', patient_subid];
corrected = [nii_folder, '/corrected.nii.gz'];
corrected = load_nii(corrected).img;
original = [nii_folder, '/original.nii.gz'];
original = load_nii(original).img;
fixed = [nii_folder, '/fixed_CT.nii.gz'];
fixed = load_nii(fixed).img;

% save dicom for original, corrected and fixed CT
for class = 1:3
    new_meta = metadata;
    
    new_meta.SeriesNumber = 1000 + class * 10;
    if class == 1
        new_meta.SeriesDescription = 'original';
        image_raw = original;
    elseif class == 2
        new_meta.SeriesDescription = 'corrected';
        image_raw = corrected;
    else
        new_meta.SeriesDescription = 'fixedCT';
        image_raw = fixed;
    end

    new_meta.PatientName = ['random_', num2str(i)];
    new_meta.SliceThickness = 2.5;
    
    image_new = image_raw;
    for z  = 1: size(image_raw,3)
        image_new(:,:,z ) = flip(flip(image_raw(:,:,z)',1),2);
    end
    
    image_final = int16(image_new - metadata.RescaleIntercept);
    
    data_save_folder = [nii_folder, '/dicoms']; mkdir(data_save_folder);
    data_save_folder = [data_save_folder, '/', new_meta.SeriesDescription]; mkdir(data_save_folder);

    for z = 1:size(image_final,3)
        new_meta.InstanceNumber = z;
        new_meta.SliceLocation = metadata.SliceLocation + 2.5 * (z-1); % pseudo-slicelocation change
    
        z_index = find(metadata.ImagePositionPatient == metadata.SliceLocation);
        new_meta.ImagePositionPatient(z_index) = new_meta.SliceLocation;
        
        % meta.ImagePositionPatient = first_plane_position + [5.0;-8; -4.0] * (z-1); % pseudo-position change (patient orientation doesn't change)
        if z<10
            filename = ['slice_00', num2str(z)];
        elseif (z>=10) && (z<100)
            filename = ['slice_0', num2str(z)];
        else
            filename = ['slice_', num2str(z)];
        end
    
        dicomwrite(image_final(:,:,z),[data_save_folder, '/', filename,'.dcm'] ,new_meta)
    end
end

end

%% without series description
for i = 1:size(patient_list,1)
patient_id = patient_list{i,"Patient_ID"}{1};
patient_subid = patient_list{i,"Patient_subID"}{1};
disp(patient_id)
disp(patient_subid)
dicom_image_folder = [data_path2, '/', patient_id, '/', patient_subid];

% load the dicom image for metadata
dicom_image_subfolder = Find_all_folders(dicom_image_folder);
dicom_image_subfolder = [dicom_image_folder, '/', dicom_image_subfolder(1,:).name];
dicom_image_files = Find_all_files(dicom_image_subfolder);
filename = [dicom_image_subfolder, '/', dicom_image_files(2,:).name];
metadata = dicominfo(filename);

% read the nii files
nii_folder = [data_path1, '/data/', patient_id, '/', patient_subid];
corrected = [nii_folder, '/corrected.nii.gz'];
corrected = load_nii(corrected).img;
original = [nii_folder, '/original.nii.gz'];
original = load_nii(original).img;
fixed = [nii_folder, '/fixed_CT.nii.gz'];
fixed = load_nii(fixed).img;

% save dicom for original, corrected and fixed CT
used_series = [];
for class = 1:3
    new_meta = metadata;
    
    new_meta.SeriesNumber = 1000 + round(rand()*500); 
    if class == 1
        used_series = [used_series, new_meta.SeriesNumber];
    elseif class > 1
        while 1==1
            aa = find(used_series == new_meta.SeriesNumber);
            if size(aa,2) == 0
                used_series = [used_series, new_meta.SeriesNumber];
                break
            end
            new_meta.SeriesNumber = 1000 + round(rand()*500);
        end
    end
    disp(new_meta.SeriesNumber)

    if class == 1
        new_meta.SeriesDescription = 'unknown';
        class_itself = 'original';
        image_raw = original;
    elseif class == 2
        new_meta.SeriesDescription = 'unknown';
        class_itself = 'corrected';
        image_raw = corrected;
    else
        new_meta.SeriesDescription = 'unknown';
        class_itself = 'fixed';
        image_raw = fixed;
    end

    new_meta.PatientName = ['Unknown'];
    new_meta.SliceThickness = 2.5;

    image_new = image_raw;
    for z  = 1: size(image_raw,3)
        image_new(:,:,z ) = flip(flip(image_raw(:,:,z)',1),2);
    end

    image_final = int16(image_new - metadata.RescaleIntercept);

    data_save_folder = [nii_folder, '/dicoms_no_description']; mkdir(data_save_folder);
    data_save_folder = [data_save_folder, '/', class_itself]; mkdir(data_save_folder);

    for z = 1:size(image_final,3)
        new_meta.InstanceNumber = z;
        new_meta.SliceLocation = metadata.SliceLocation + 2.5 * (z-1); % pseudo-slicelocation change

        z_index = find(metadata.ImagePositionPatient == metadata.SliceLocation);
        new_meta.ImagePositionPatient(z_index) = new_meta.SliceLocation;

        % meta.ImagePositionPatient = first_plane_position + [5.0;-8; -4.0] * (z-1); % pseudo-position change (patient orientation doesn't change)
        if z<10
            filename = ['slice_00', num2str(z)];
        elseif (z>=10) && (z<100)
            filename = ['slice_0', num2str(z)];
        else
            filename = ['slice_', num2str(z)];
        end

        dicomwrite(image_final(:,:,z),[data_save_folder, '/', filename,'.dcm'] ,new_meta)
    end
end

end