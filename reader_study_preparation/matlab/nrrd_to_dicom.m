clear all; close all; clc;
code_path = '/Users/zhennongchen/Documents/GitHub/CMR_HFpEF_Analysis/matlab';
addpath(genpath(code_path));
data_path = '/Volumes/TOSHIBA_4TB/MGH/HFpEF_zhennong/nnrd';
save_path = '/Volumes/TOSHIBA_4TB/MGH/HFpEF_zhennong/dicoms';
info = load('/Users/zhennongchen/Documents/Zhennong_CMR_HFpEF/Data/HFpEF/infoData.mat');
info = info.infoData;

ID_list = [];
timeframe_nums = [];
dimension_error = [];
[folders] = Find_all_folders(data_path);
%% find cases
excel = readtable('/Users/zhennongchen/Documents/Zhennong_CMR_HFpEF/Data/HFpEF/Patient_list/Important_HFpEF_Patient_list_unique_patient.xlsx');
ids = excel.OurID;

%% load case
for idsids = 1:size(ids,1)
    i = ids(idsids);
    % metadata
    meta = info{i};

    patient_id = folders(i).name;
    disp(patient_id);
    ID_list = [ID_list; patient_id];
    patient_folder = [folders(i).folder, '/', folders(i).name];
    patient_files = Find_all_files(patient_folder);
    
    % assert time frames = 25 
    timeframe_nums = [timeframe_nums; size(patient_files,1)];
%     if size(patient_files,1) ~= 25
%         disp(['this case does not have 25 time frame'])
%         timeframe_error = [timeframe_error;"error"];
%         continue
%     else
%         timeframe_error = [timeframe_error;""];
%     end
    
    patient_files =  Sort_time_frame(Find_all_files(patient_folder), 'e'); 
    
    % load all timeframes
    for t = 1:size(patient_files,1)
        filename = [patient_folder, '/', convertStringsToChars(patient_files(t,:))];
        [ii,~] = nrrdread(filename);
        I{t} = ii;
    end

    % assert the dimension of each time frame is the same
    dimension_list = [];
    for t = 1:size(patient_files,1)
        ii = I{t};
        dimension_list = [dimension_list; size(ii)];
    end

    if all(ismember(dimension_list, dimension_list(1, :), 'rows')) == 0
        disp(['not consistent dimension across time frames!'])
        dimension_error = [dimension_error;"error"];
        continue
    else
        dimension_error = [dimension_error;""];
    end
    
    if isfile([save_path, '/', patient_id,'/functions/slice_1/tf_001.dcm'])  == 1
        disp(['already done!'])
        continue
    end
    
    % save dicom (each image volume)
    save_folder = [save_path, '/', patient_id]; mkdir(save_folder)
    save_folder_volumes = [save_folder, '/', 'volumes']; mkdir(save_folder_volumes);
    
    % define the first plane patient position:
    first_plane_position = meta.ImagePositionPatient;
    
    for tf = 1: size(I,2)
        save_folder_volume = [save_folder_volumes,'/','timeframe_', num2str(tf)]; mkdir(save_folder_volume);
        img = I{tf};

        meta.SeriesNumber = 1000 + tf * 10;
        meta.SeriesDescription = ['image_volume_timeframe_', num2str(tf)];

        for z = 1:size(img,3)
            meta.InstanceNumber = z;
            meta.SliceLocation = (z-1) * 8; % pseudo-slicelocation change
            meta.ImagePositionPatient = first_plane_position + [5.0;-8; -4.0] * (z-1); % pseudo-position change (patient orientation doesn't change)
            if z<10
                filename = ['slice_00', num2str(z)];
            elseif (z>=10) && (z<100)
                filename = ['slice_0', num2str(z)];
            else
                filename = ['slice_', num2str(z)];
            end

            dicomwrite(img(:,:,z),[save_folder_volume, '/', filename,'.dcm'] ,meta)
        end
    end


    % save dicom (each slice across one heart beat)
    save_folder = [save_path, '/', patient_id]; mkdir(save_folder)
    save_folder_functions = [save_folder, '/', 'functions']; mkdir(save_folder_functions);

    img = I{1};

    for z = 1:size(img,3)
        save_folder_function_slice = [save_folder_functions,'/','slice_', num2str(z)]; mkdir(save_folder_function_slice);

        meta.SeriesNumber = 2000 + z * 10;
        meta.SeriesDescription = ['function_slice_', num2str(z)];
        meta.ImagePositionPatient = first_plane_position + [5.0;-8; -4.0] * (z-1);
        
        for tf = 1:size(I,2)
            meta.InstanceNumber = tf;
            meta.SliceLocation = (tf-1) * 8;
            if tf<10
                filename = ['tf_00', num2str(tf)];
            elseif (tf>=10) && (tf<100)
                filename = ['tf_0', num2str(tf)];
            else
                filename = ['tf_', num2str(tf)];
            end

            img = I{tf};
            dicomwrite(img(:,:,z),[save_folder_function_slice, '/', filename,'.dcm'] ,meta)
        end
    end
end
%
table_data = table(ID_list, timeframe_nums, dimension_error);
filename = 'nrrd_to_dicom_notes.xlsx';
writetable(table_data, filename);

    