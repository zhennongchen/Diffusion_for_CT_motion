%% Preparation: Data Organization
clear all; close all; clc;
addpath(genpath('/Users/zhennongchen/Documents/GitHub/CMR_HFpEF_Analysis/matlab'));
addpath(genpath('/Users/zhennongchen/Documents/GitHub/Volume_Rendering_by_DL/matlab'));

main_path = '/Users/zhennongchen/Documents/Zhennong_CMR_HFpEF/Data/Sunny_Brooks/examples';
%% Load
id = 'SC-HYP-08';
lax_img = load_nii([main_path,'/',id,'/LAX_img_crop.nii.gz']);
lax_img = flip(lax_img.img,2);

lax_img_contours = load_nii([main_path,'/',id,'/LAX_img_w_contours_pred.nii.gz']);
lax_img_contours = flip(lax_img_contours.img,2);
%%
[endo_row, endo_col] = find(lax_img_contours == 2500);
[epi_row, epi_col] = find(lax_img_contours == 2700);
%%
lax_img_I = Turn_data_into_greyscale(lax_img,135,270);     
figure(1)
imshow(lax_img_I,'InitialMagnification', 'fit');
%% move lax_img
translatedImage = imtranslate(lax_img_I, [0,8],'FillValues',min(lax_img_I(:)));

for i = 1:size(endo_col,1)
    if endo_row(i) ~= 77
        translatedImage(endo_row(i), endo_col(i)) = 0.9;
    end
end
for i = 1:size(epi_col,1)
    if epi_row(i) ~= 77
        translatedImage(epi_row(i), epi_col(i)) = 1.3;
    end
end
figure(1)
a = translatedImage(9:end,:);
imshow(a,'InitialMagnification', 'fit');



