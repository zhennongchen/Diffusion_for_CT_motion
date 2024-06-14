%% Preparation: Data Organization
clear all; close all; clc;
code_path = '/Users/zhennongchen/Documents/GitHub/CMR_HFpEF_Analysis/matlab';
addpath(genpath(code_path));
addpath(genpath('/Users/zhennongchen/Documents/GitHub/Volume_Rendering_by_DL/matlab/'));
data_path = '/Users/zhennongchen/Documents/Zhennong_CMR_HFpEF/Data/Stony_Brooks/Surface_points';

%% load points
case_id1 = 'SC-N-02';
case_id2 = 'SC-N-03';


case1 = load([data_path,'/',case_id1,'_ED.mat']); Mesh1 = case1.Mesh;
case2 = load([data_path,'/',case_id2,'_ED.mat']); Mesh2 = case2.Mesh;
clear case1 case 2

%% opt parameters
% CPD Parameters
opts.corresp = 1;
opts.normalize = 1;
opts.max_it = 1500;
opts.tol = 1e-5;
opts.viz = 0;
opts.method = 'nonrigid_lowrank';
opts.fgt = 0;
opts.eigfgt = 0;
opts.numeig = 100;
opts.outliers = 0.05;
opts.beta = 2;
opts.lambda = 3;

% %% CPD (directly on point cloud)
% point_cloud1 = sortrows(unique(int16(Mesh1.point_cloud),'rows'),3);
% point_cloud2 = sortrows(unique(int16(Mesh2.point_cloud),'rows'),3);
% [T,C] = cpd_register(point_cloud1,point_cloud2,opts);
% CPD = T.Y; 
% %% Validate Results
% % theory: A point on nth slice in Heart i, [xn,i, yn,i, zn,i] can only be registered to a point on the same slice in Heart j, [xn,j, yn,j, zn,j]
% 
% for i = 1:size(Mesh2.I,3)
%     count = 0;total = 0;
%     idx = point_cloud2(:,3) == i;
%     CPD_query = CPD(idx,:);
%     
%     
%     buffer = 1.20;
%     for j = 1:size(CPD_query,1)
%         total = total+1;
%         if (CPD_query(j,3) < i-buffer) || (CPD_query(j,3) > i+buffer)
%             count = count+1;
%         end
%     end
%     
%     disp(['after CPD, ', num2str(count),' out of ',num2str(total), ' points on slice ', num2str(i), ' is on wrong slice']);
% end
%     
%     
% %% show results
% im_cpd = zeros(size(Mesh2.I));
% 
% % point_cloud1_int = int16(point_cloud1);
% 
% ind = sub2ind(size(Mesh2.I),point_cloud1(:,1),point_cloud1(:,2),point_cloud1(:,3));
% im_cpd(ind) = 10;
% 
% % point_cloud2_int = int16(point_cloud2);
% ind = sub2ind(size(Mesh2.I),point_cloud2(:,1),point_cloud2(:,2),point_cloud2(:,3));
% im_cpd(ind) = 5;
% 
% cpd_int = int16(CPD);
% ind = sub2ind(size(Mesh2.I),cpd_int(:,1),cpd_int(:,2),cpd_int(:,3));
% im_cpd(ind) = 20;
% %%
% slice = 10;
% imagesc(im_cpd(:,:,slice))


%% CPD (use surface mesh)
[T,C] = cpd_register(Mesh1.vertices,Mesh2.vertices,opts);
CPD = T.Y;
%% Show results
im_template = zeros(size(Mesh1.I));
im_cpd_template = zeros(size(Mesh1.I));
im_cpd_LV = zeros(size(Mesh1.I));

% im_template:
v = Mesh1.vertices;
v = sortrows(unique(v,'rows'),3);
im_template = zeros(size(Mesh1.I));
ind = sub2ind(size(Mesh1.I),v(:,1),v(:,2),v(:,3));
im_template(ind) = 1;
ind2 = sub2ind(size(Mesh1.I),Mesh1.LV_points(:,2),Mesh1.LV_points(:,1),Mesh1.LV_points(:,3));
im_template(ind2) = 2;
slice = 3
imagesc(im_template(:,:,slice));
%%






CPD_int = int16(CPD);
CPD_int = sortrows(unique(int16(CPD_int),'rows'),3);
l = CPD_int(:,3) <= size(Mesh1.I,3);
CPD_int = CPD_int(l,:);

ind = sub2ind(size(Mesh1.I),CPD_int(:,1),CPD_int(:,2),CPD_int(:,3));
im_cpd(ind) = 1;
im_cpd_template(ind) = 1;
im_cpd_LV(ind) = 2;
%% template
v = Mesh1.vertices;
v = sortrows(unique(v,'rows'),3);
im_template = zeros(size(Mesh1.I));
ind = sub2ind(size(Mesh1.I),v(:,1),v(:,2),v(:,3));
im_template(ind) = 1;
im_cpd_template(ind) = 2;


ind2 = sub2ind(size(Mesh1.I),Mesh1.LV_points(:,2),Mesh1.LV_points(:,1),Mesh1.LV_points(:,3));
im_cpd_LV(ind2) = 1;
%%
slice = 7;
figure()
imagesc(im_cpd_template(:,:,slice))




