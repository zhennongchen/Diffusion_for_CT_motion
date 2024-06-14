%%% this script is for those patients with multiple folders for cine SAX
%%% data in the raw data
clear all; close all; clc;
code_path = '/Users/zhennongchen/Documents/GitHub/CMR_HFpEF_Analysis/matlab';
addpath(genpath(code_path));
data_path = '/Volumes/Camca/home/ZC/AS_CMR_data/dicoms_img';
save_path = '/Volumes/Camca/home/ZC/AS_CMR_data/dicoms_img';
% info = load('/Users/zhennongchen/Documents/Zhennong_CMR_HFpEF/Data/HFpEF/infoData.mat');
% info = info.infoData;

ID_list = [];
timeframe_nums = [];
dimension_error = [];
[folders] = Find_all_folders(data_path);
%% find cases
excel = readtable('/Users/zhennongchen/Documents/Zhennong_CMR_HFpEF/Data/HFpEF/Patient_list/Important_HFpEF_Patient_list_unique_patient.xlsx');
ids = excel.OurID;

%% load case
for i = 20%[6,11,40,42]
    patient_id = folders(i).name;
    disp(patient_id);
    patient_folder_main = [folders(i).folder, '/', folders(i).name,'/volumes'];
    patient_folder_tfs = Find_all_folders(patient_folder_main);
    
    for tf  = 1 : size(patient_folder_tfs,1)
        disp(['tf ', num2str(tf)])
        patient_folder_tf = [patient_folder_main,'/','timeframe_',num2str(tf)];
        patient_files = Find_all_files(patient_folder_tf);
        new_save_folder1 = [folders(i).folder, '/', folders(i).name,'/volumes_new'];mkdir(new_save_folder1)
        new_save_folder = [folders(i).folder, '/', folders(i).name,'/volumes_new/timeframe_', num2str(tf)];mkdir(new_save_folder)

        for t = 1:size(patient_files,1)
            filename = [patient_folder_tf, '/', patient_files(t,:).name];
         
            img = dicomread(filename); 
            if t == 1
                uniform_metadata = dicominfo(filename);
            end
    
            metadata = dicominfo(filename);
            slice_location = metadata.SliceLocation;
            patient_position = metadata.ImagePositionPatient;
            patient_orientation = metadata.ImageOrientationPatient;
    
            metadata = uniform_metadata;
            metadata.slice_location = slice_location;
            metadata.ImagePositionPatient = patient_position;
            metadata.ImageOrientationPatient = patient_orientation;
    
            dicomwrite(img,[new_save_folder, '/', patient_files(t,:).name] ,metadata)
        end
    end
end