%% Step 3
%% Load path
addpath('function');

% load ProcessingPath.mat
ProcessingPath = '../';
% % Default
Ref_Frame = 12;

list = dir([ProcessingPath,'/Processing/ID_*']);

%%
% try
    for ii = 10%1:size(list,1)
        
        DataPath = [list(ii).folder,'/',list(ii).name];
        
        Framelist = dir([DataPath,'/Org3D_frame*']);
        
        [img_ref, meta_ref] = nrrdread([DataPath,'/Img3D_frame',num2str(Ref_Frame,'%.4d'),'.nrrd']);
        
        load([DataPath,'/Seg_frame',num2str(Ref_Frame,'%.4d')]);
        nrrdWriter([DataPath,'/Seg_frame',num2str(Ref_Frame,'%.4d'),'.nrrd'],img_seg,Res,[0,0,0],'raw');
        
        load([DataPath,'/mask.mat']);
        %%
        for it = 1:length(Framelist)
            
            %% Upsample and calculate Deformation
            if it~= Ref_Frame
                
                [imgMR, meta] = nrrdread([DataPath,'/Org3D_frame',num2str(it),'.nrrd']);
                
                [C] = str2double(strsplit(meta.spacedirections,{'(',',',')',' '}));
                T = C(2:10);
                
                % extract resolution information from it
                dx = sqrt(T(1)^2+T(2)^2+T(3)^2);
                dy = sqrt(T(4)^2+T(5)^2+T(6)^2);
                dz = sqrt(T(7)^2+T(8)^2+T(9)^2);
                
                [C] = str2double(strsplit(meta.spaceorigin,{'(',',',')',' '}));
                Offset = C(2:4);
                
                % unit direction
                T = reshape(T./[dx dx dx dy dy dy dz dz dz],3,3);
                
                tform = zeros(4,4);
                tform(1:3,1:3) = T';
                tform(4,1:3) = -Offset;
                tform(4,4) = 1;
                
                tform_translate = affine3d(tform);
                imgMR = imwarp(imgMR,imref3d(size(imgMR)),tform_translate);                
                
                
                Res = [dx dy dz];
                [nx, ny, nz] = size(imgMR);
                
                ox = single([0:nx-1] - (nx-1.0)/2.0)*dx;
                oy = single([0:ny-1] - (ny-1.0)/2.0)*dy;
                oz = single([0:nz-1] - (nz-1.0)/2.0)*dz;
                
                [oyy, oxx, ozz] = meshgrid(oy, ox, oz);
                
                % upsample of mask
                tnx = 256;
                tny = 256;
                tnz = 256;
                
                tdx = 1; % mm
                tdy = 1;
                tdz = 1;
                
                x = single([0:tnx-1] - (tnx-1.0)/2.0)*tdx;
                y = single([0:tny-1] - (tny-1.0)/2.0)*tdy;
                z = single([0:tnz-1] - (tnz-1.0)/2.0)*tdz;
                
                [yy, xx, zz] = meshgrid(y, x, z);
                img = interp3(oyy, oxx, ozz, single(imgMR), yy, xx, zz);
                img(isnan(img)) = 0;
                
                nrrdWriter([DataPath,'/Img3D_frame',num2str(it,'%.4d'),'.nrrd'],img,[tdx, tdy, tdz],[ 0 0 0],'raw');
                
                
                [D,deformed_source] = imregdemons((single(img_ref)),(single(img)),[500 300 50],'AccumulatedFieldSmoothing',1.0);
%                 [D,deformed_source] = imregdemons((single(img_ref)),(single(img)),[1 1 1],'AccumulatedFieldSmoothing',1.0);
                
                img_outer_frame = int16(imwarp(img_outer,D));
                img_inner_frame = int16(imwarp(img_inner,D));
                img_mark_frame = int16(imwarp(img_mark,D));
                
                img_seg_frame = Seg17ks(img_outer_frame,img_inner_frame,img_mark_frame,[tdx,tdy,tdz]);
                
%                 img_seg_frame = zeros(size(img_seg),'int16');
%                 for iseg = 1:17
%                     tmp_seg = zeros(size(img_seg),'single');
%                     tmp_seg(img_seg == iseg) = 1;
% %                     img_seg_frame = img_seg_frame + iseg*int16(imwarp(tmp_seg,D));
%                     img_seg_frame = img_seg_frame + iseg*int16(imwarp(tmp_seg,D));
%                 end
                

                
                opacity = 0.3;
                Slice = 128;
                img_fuse = fusemripet(img(:,:,Slice), img_seg_frame(:,:,Slice), opacity, 1.7);
%                 img_fuse2 = fusemripet(squeeze(img(:,Slice,:)), squeeze(img_seg_frame(:,Slice,:)), opacity, 1.7);
%                 img_fuse3 = fusemripet(squeeze(img(Slice,:,:)), squeeze(img_seg_frame(Slice,:,:)), opacity, 1.7);
                figure(33); imagesc(img_fuse); axis off; axis equal; title(['Frame: ',num2str(it)]);
%                 figure(33); 
%                 subplot(311);imagesc(img_fuse1); axis off; axis equal; title(['Frame: ',num2str(it)]);
%                 subplot(312);imagesc(img_fuse2); axis off; axis equal; title(['Frame: ',num2str(it)]);
%                 subplot(313);imagesc(img_fuse3); axis off; axis equal; title(['Frame: ',num2str(it)]);
%                 tmp_frame{it} = getframe;
                
%                 figure(34); 
%                 subplot(321);imagesc(img_seg_frame(:,:,Slice)); axis off; axis equal; title(['Frame: ',num2str(it)]); colormap jet;
%                 subplot(323);imagesc(squeeze(img_seg_frame(:,Slice,:))'); axis off; axis equal; title(['Frame: ',num2str(it)]);
%                 subplot(325);imagesc(squeeze(img_seg_frame(Slice,:,:))'); axis off; axis equal; title(['Frame: ',num2str(it)]);
%                 subplot(322);imagesc(img(:,:,Slice)); axis off; axis equal; title(['Frame: ',num2str(it)]);
%                 subplot(324);imagesc(squeeze(img(:,Slice,:))'); axis off; axis equal; title(['Frame: ',num2str(it)]);
%                 subplot(326);imagesc(squeeze(img(Slice,:,:))'); axis off; axis equal; title(['Frame: ',num2str(it)]);
                
                
                
                
                pause(0.1);
                
                nrrdWriter([DataPath,'/Seg_frame',num2str(it,'%.4d'),'.nrrd'],img_seg_frame,[tdx, tdy, tdz],[ 0 0 0],'raw');
                                
                
            else
                opacity = 0.4;
                
                img_fuse = fusemripet(img(:,:,Slice), img_seg(:,:,Slice), opacity, 1.1);
                figure(33); imagesc(img_fuse); axis off; axis equal; title(['Frame: ',num2str(Ref_Frame)]);
%                 tmp_frame{it} = getframe;
            end            
            
            
        end
        %%
%         v= VideoWriter([DataPath,'/MotionSeg'],'Motion JPEG AVI');


        v= VideoWriter([DataPath,'/MovieSeg'],'Motion JPEG AVI');
        v.FrameRate = 5;
        open(v);
        for iit = 1:length(Framelist)
            img = nrrdread([DataPath,'/Img3D_frame',num2str(iit,'%.4d'),'.nrrd']);
            img_seg = nrrdread([DataPath,'/Seg_frame',num2str(iit,'%.4d'),'.nrrd']);
            Slice = 120;
            img_fuse = fusemripet(img(:,:,Slice), img_seg(:,:,Slice), opacity, 1.1);
            figure(33); imagesc(img_fuse); axis off; axis equal; title(['Frame: ',num2str(iit)]);
            pause(0.1);
            writeVideo(v,getframe);
        end
        close(v);
    end
    
% catch
%     if (isempty(list) == 1)
%         disp('No available data exists')
%     else
%         disp(['Error with processing: ']);
%     end
% end
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
