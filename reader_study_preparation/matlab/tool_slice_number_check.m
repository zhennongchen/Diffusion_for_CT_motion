%% Preparation: Data Organization
clear all; close all; clc;
code_path = '/Users/zhennongchen/Documents/GitHub/CMR_HFpEF_Analysis/matlab';
addpath(genpath(code_path));

data_path = '/Users/zhennongchen/Documents/Zhennong_CMR_HFpEF/Data/HFpEF/';
dropbox_path = '/Users/zhennongchen/Dropbox (Partners HealthCare)/CMR/Processing/ForMaskDrawing/';
%%
table_file = [data_path, 'AccessionNumber_ID_list_HFpEF.xlsx'];
Table = readtable(table_file,'range','A:B');
%%
slice_num_list = [];
ID_list = [];
non_blank_slice_num_list = [];
blank_slice_num_list = [];
blank_slice_position_list = [];
suggestions = [];


for i = 1: size(Table,1)
    ID = Table.OurID(i); ID = ID{1};
    
    ID_num = str2num(ID);
    if ID_num<10
        file = ['ID_000', ID, '_frame12.nrrd'];
        id = ['ID_000', ID];
    elseif (ID_num>=10) && (ID_num < 100)
        file = ['ID_00', ID, '_frame12.nrrd'];
        id = ['ID_00', ID];
    elseif (ID_num>=100) && (ID_num < 1000)
        file = ['ID_0', ID, '_frame12.nrrd'];
        id = ['ID_0', ID];
    else 
        file = ['ID_', ID, '_frame12.nrrd'];
        id = ['ID_', ID];
    end
    ID_list = [ID_list;id];
    [I, meta] = nrrdread([dropbox_path, file]);
    slice_num_list = [slice_num_list; size(I,3)];
    
    % find non-blank slices and blank/zero slices
    % first get the pixel value summation in each z slices
    sums = squeeze(sum(sum(double(I),1),2));
   
    % second, let's check the position of blank slices
    % 1. if the blank slices are all either at the beginning or at the end,
    % we can easily remove them
    
    % 2. if the blank slices are not all at the beginning or at the end
    % (meaning it's between two non-zero slices), it indicates the missing
    % data and this case should be suggested for exclusion.
    
    nz = find(sums ~= 0);
    z = find(sums == 0);
    
    non_blank_slice_num_list = [non_blank_slice_num_list; size(nz,1)];
    blank_slice_num_list = [blank_slice_num_list; size(z,1)];
    
    if isempty(z)
        blank_slice_position_list = [blank_slice_position_list; " "];
        suggestions = [suggestions; " "];
    elseif all(diff(nz) == 1)
        blank_slice_position_list = [blank_slice_position_list; "blank slices at two ends"];
        suggestions = [suggestions; "remove blank slices before AI steps"];
    else
        blank_slice_position_list = [blank_slice_position_list; "blank slices in the middle"];
        suggestions = [suggestions; "Exclude this case!"];
    end
   
    
end
%%
table_data = table(ID_list, slice_num_list, non_blank_slice_num_list, blank_slice_num_list, blank_slice_position_list,suggestions );
filename = 'slice_numbers.xlsx';
writetable(table_data, filename);
