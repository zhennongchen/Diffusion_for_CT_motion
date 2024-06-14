%% Load nii:
clear all
cd '/Users/zhennongchen/Documents/Zhennong_PHD_Brightness Calibration Application/Dataset/Heartflow_tests_2016/Cases/ucsd_case_003/Segmentation';
data = load_nii('case3_threshold220.nii.gz');
cd '/Users/zhennongchen/Documents/GitHub/thesis_matlab/Brightness Calibration Application/NIfTI image processing'
%% pixel spacing:
dx = data.hdr.dime.pixdim(2);
dy = data.hdr.dime.pixdim(3);
dz = data.hdr.dime.pixdim(4);
%% Extracting only the vessel with tag = 1
seg = zeros(size(data.img));
seg(data.img==1) =1;
%% Direction matrix Transformation from .nii to .mat
seg = permute(seg,[2 1 3]);
seg = flip(seg,1);
seg = flip(seg,2);
% 
%% Import the corresponding DICOM image
cd '/Users/zhennongchen/Documents/Zhennong_PHD_Brightness Calibration Application/Dataset/Heartflow_tests_2016/Cases/ucsd_case_003/CT/74-84 303'
files=dir('*.dcm');
Image=[];
info = dicominfo(files(1,1).name);
for l=1:numel(files)
    img=dicomread(files(l,1).name);
    img = info.RescaleSlope.*img + info.RescaleIntercept;
   Image(:,:,l) = img;
end
pixel_size=[];
for l=1:numel(files)
info=dicominfo(files(l).name);pixel_size(l)=info.PixelSpacing(1);
end
pixel_size=pixel_size(1);
cd '/Users/zhennongchen/Documents/GitHub/thesis_matlab/Brightness Calibration Application/NIfTI image processing'
%% Extract vessel from DICOM
vessel = [];
z_size = size(Image,3);
for i = 1:z_size
    vessel(:,:,i) = Image(:,:,i);
end
vessel = int16(vessel);
[x,y,z]=ind2sub(size(seg),find(seg~=1));
for i = 1: size(x,1)
    vessel(x(i),y(i),z_size-(z(i)-1))= -1024 ; % convert the z in ITK to z in CCTA
end

%% Save vessel
%%export as Dicom
cd '/Users/zhennongchen/Documents/Zhennong_PHD_Brightness Calibration Application/Dataset/Heartflow_tests_2016/Cases/ucsd_case_003/CT/74-84 303'
for i=1:numel(files)
X=Name_new_file('LCX_',i,'.dcm',4);
metadata=dicominfo(files(i).name);
metadata.SeriesDescription = 'LCX';
%metadata.RescaleIntercept = 0;
metadata.SeriesInstanceUID = '2.16.840.1.113669.632.21.56297304.1903900265.27246536812217449.2';
dicomwrite(vessel(:,:,i),X,metadata);
end
% move to new directory
mkdir LCX
movefile LCX* LCX
%%
