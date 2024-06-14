clear
root = 'C:\Users\37908\Documents\MATLAB\Longitudinal\1106_time_test\';

epi_endo = 'ENDO';
lax_sax = 'SAX';
png_root = [root,'converted_png\',lax_sax,'\'];
msk_root = [root,'mask\',epi_endo,'\'];

listing = dir(png_root);
dx = 1;dy = 1; dz = 1;
for i = 3:length(listing)

	folder = listing(i).name;
	
	png_path = [png_root,folder,'\'];
	listing2 = dir(png_path);
	framenum = length(listing2)-2;
% 	h = figure('visible','off','rend','painters','pos',[10 10 10000 400]);
	for t = 3:length(listing2)
		t
		caseName = listing2(t).name;
		IMG = histeq(imread([png_path,caseName]));
		IMG(:,:,2) = IMG;
		MSK = imread([msk_root,folder,'\',caseName(1:end-4),'_mask.png']);
		MSK(:,:,2) = MSK;
		out_path = strrep(png_path,'converted_png',['nrrd\',epi_endo,'\']);
		if(~exist(out_path,'dir'))
			mkdir(out_path);
		end
		nrrdWriter([out_path,caseName(1:end-4),'.nrrd'],IMG,[dx, dy, dz],[ 0 0 0],'raw');
		nrrdWriter([out_path,caseName(1:end-4),'_mask.nrrd'],MSK,[dx, dy, dz],[ 0 0 0],'raw');
	end
end

%%
root = './manually_modify_contour/';
img_name = 'Image_0003.png';
msk_name = 'Image_0003_mask.png';

IMG = imread([root,img_name]);
MSK = imread([root,msk_name]);
% for t = 1:10
%    IMG(:,:,t) = IMG(:,:,1); 
%    MSK(:,:,t) = MSK(:,:,1);
% end

dx = 1;dy = 1; dz = 1;

nrrdWriter(['IMG','.nrrd'],IMG,[dx, dy, dz],[ 0 0 0],'raw');
nrrdWriter(['MSK','.nrrd'],MSK,[dx, dy, dz],[ 0 0 0],'raw');


%%
[X, meta] = nrrdread('F:\MSK_3d-label.nrrd');
X = X(:,:,2);
result = X(:,:,1);
figure(1729); imshow(X,[])
result = uint16(result);
imagetype = 'png';
imwrite(result,'adjusted_msk2.png', imagetype);









%%
[X, meta] = nrrdread('F:\MSK_3d-label.nrrd');
X = uint16(X(:,:,2));
figure(1729); imshow(X,[])

imwrite(X,[],'ground_truth.png', 'png');
