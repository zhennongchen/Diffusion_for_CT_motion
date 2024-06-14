%% Step 2 is working after drawing masks
%% Load path

% load ProcessingPath.mat
addpath('function');
% load ProcessingPath.mat
ProcessingPath = 'H:/CMR';

% % Default
MaskPath = [ProcessingPath,'/Processing/ForMaskDrawing'];
Mask_LV = dir([MaskPath,'/*outer*']);
Mask_sub = dir([MaskPath,'/*inner*']);
Mask_LM = dir([MaskPath,'/*landmark*']);

Ref_Frame = 12;

%%
% try
    for ii = 1:size(Mask_LV,1)
        name = Mask_LV(ii).name;
        tmp = strsplit(name,'_');
        nid = str2num(tmp{2});
        
        DataPath = [ProcessingPath,'/Processing','/ID_',num2str(nid,'%.4d')];
        
        
        %% Load mask
        [mask, meta] = nrrdread([Mask_LV(ii).folder,'/',Mask_LV(ii).name]);
        mask(mask>0) = 1;
        [mask_sub, meta] = nrrdread([Mask_sub(ii).folder,'/',Mask_sub(ii).name]);
        mask_sub(mask_sub>0) = 1;
        
        [mark, meta] = nrrdread([Mask_LM(ii).folder,'/',Mask_LM(ii).name]);
        mark(mark>0) = 1;
        
%         mask = max(mask - mask_sub,0);
        
        figure(1); imagesc(mask(:,:,round(end/2))); axis off; axis equal; colormap gray;
        
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
        mask = imwarp(mask,imref3d(size(mask)),tform_translate);
        mask_sub = imwarp(mask_sub,imref3d(size(mask_sub)),tform_translate);
        mark = imwarp(mark,imref3d(size(mark)),tform_translate);
        
        
                
        Res = [dx dy dz];
        
        [nx, ny, nz] = size(mask);
        
        
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
        
        img_inner = interp3(oyy, oxx, ozz, single(mask_sub), yy, xx, zz);
        img_inner(isnan(img_inner)) = 0;
        img_inner(img_inner>0.5) = 1;
        img_inner(img_inner<=0.5) = 0;
        
         
        
        
        img_outer  = interp3(oyy, oxx, ozz, single(mask), yy, xx, zz);
        img_outer(isnan(img_outer)) = 0;
        img_outer(img_outer>0.5) = 1;
        img_outer(img_outer<=0.5) = 0;
        
        img_mark  = interp3(oyy, oxx, ozz, single(mark), yy, xx, zz);
        img_mark(isnan(img_mark)) = 0;
        img_mark(img_mark>0.5) = 1;
        img_mark(img_mark<=0.5) = 0;
        
        
        save('-v7.3',[DataPath,'/mask.mat'],'img_outer','img_inner','img_mark');
        
        img_seg = Seg17ks(img_outer,img_inner,img_mark,[tdx,tdy,tdz]);
%         
%         % % land mark -> center estimation
%         L = bwlabeln(mark);
%         for im = 1:2
%             id = find(L==im);
%             id_mark(im) = id(1);
%         end
%          
%         LM1 = [ oxx(id_mark(1)), oyy(id_mark(1)), ozz(id_mark(1))];
%         LM1_id = (LM1./[dx, dy, dz] + [(nx-1.0)/2.0, (ny-1.0)/2.0, (nz-1.0)/2.0])+1;
%         LM2 = [ oxx(id_mark(2)), oyy(id_mark(2)), ozz(id_mark(2))];
%         LM2_id = (LM2./[dx, dy, dz] + [(nx-1.0)/2.0, (ny-1.0)/2.0, (nz-1.0)/2.0])+1;
% 
%         tmp = mask(:,:,LM1_id(3));
%         CM_id = [ sum(sum(tmp,2)'.*[1:nx])/sum(tmp(:)), sum(sum(tmp,1).*[1:ny])/sum(tmp(:))]; 
%         LM3= (LM1+LM2)/2;
%         LM3_id = LM3./[dx, dy, dz] + [(nx-1.0)/2.0, (ny-1.0)/2.0, (nz-1.0)/2.0]+1;       
%         
%         CM = (CM_id-1- [(nx-1.0)/2.0,(ny-1.0)/2.0]).*[dx,dy]; 
%         
%         
%           
%         
%         
%         zzz = squeeze(sum(sum(img_inner,1),2));
%         
%         z_max_id = max(find(zzz>0));
%         z_min_id = min(find(zzz>0));
%         z_max = (z_max_id-1- (tnz-1.0)/2.0).*[tdz];
%         z_min = (z_min_id-1- (tnz-1.0)/2.0).*[tdz]; 
%         
%         id = find(max(img_outer-img_inner,0)>0);
%         
%         img_seg = zeros(tnx, tny, tnz, 'single');
%         
%         theta1 = mod(atan2(LM1(1)-CM(1),LM1(2)-CM(2))/pi*180,360);
%         theta2 = mod(atan2(LM2(1)-CM(1),LM2(2)-CM(2))/pi*180,360);
%         theta3 = mod(atan2(LM3(1)-CM(1),LM3(2)-CM(2))/pi*180,360);
%         
%         theta1n = mod(theta1+180,360);
%         theta2n = mod(theta2+180,360);
%         theta3n = mod(theta3+180,360);
%         
%         cen6 = [(theta1+theta2n)/2, (theta1+theta3)/2, (theta2+theta3)/2, (theta2+theta1n)/2, (theta1n+theta3n)/2, (theta2n+theta3n)/2];
%         if abs(theta1-theta2n)>180 cen6(1) = cen6(1)+180; end
%         if abs(theta1-theta3)>180 cen6(2) = cen6(2)+180; end
%         if abs(theta2-theta3)>180 cen6(3) = cen6(3)+180; end
%         if abs(theta2-theta1n)>180 cen6(4) = cen6(4)+180; end
%         if abs(theta1n-theta3n)>180 cen6(5) = cen6(5)+180; end
%         if abs(theta2n-theta3n)>180 cen6(6) = cen6(6)+180; end
%         cen6 = mod(cen6,360);
%         
%                 
%         cen4 = mod([theta3+90, theta3, theta3-90, theta3+180],360);
%         
%         
%                 
%         for iid = 1:length(id)
%             pos = [xx(id(iid)), yy(id(iid)), zz(id(iid))];
%             
%             theta = mod(atan2(pos(1)-CM(1),pos(2)-CM(2))/pi*180,360);
%             
%             if pos(3)<z_min
%                 img_seg(id(iid)) = 17;
%             elseif pos(3)>=z_min && pos(3)<z_min+1/3*(z_max-z_min) % seg 13~16
%                                 
%                 angle_diff = abs(theta - cen4);
%                 for j=1:4 
%                     if angle_diff(j)>180 
%                         angle_diff(j) = abs(angle_diff(j)-360); 
%                     end
%                 end
%                 
%                 seg = find(angle_diff == min(angle_diff));
%                 seg = seg(1)+12;
%                 
%                 
%                 img_seg(id(iid)) = seg;
%                 
%             elseif pos(3)>=z_min+1/3*(z_max-z_min) && pos(3)<z_min+2/3*(z_max-z_min) % seg 7~12
%                 
%                 angle_diff = abs(theta - cen6);
%                 for j=1:6 
%                     if angle_diff(j)>180 
%                         angle_diff(j) = abs(angle_diff(j)-360); 
%                     end
%                 end
%                 
%                 seg = find(angle_diff == min(angle_diff));
%                 seg = seg(1)+6;                
%                 
%                 img_seg(id(iid)) = seg;
%             
%             else % seg 1~6
%                 angle_diff = abs(theta - cen6);
%                 for j=1:6 
%                     if angle_diff(j)>180 
%                         angle_diff(j) = abs(angle_diff(j)-360); 
%                     end
%                 end
%                 
%                 seg = find(angle_diff == min(angle_diff));
%                 seg = seg(1);
%                                 
%                 img_seg(id(iid)) = seg;
%         
%             end
%         end
        
        Dim = [tnx, tny, tnz];
        Res = [tdx, tdy, tdz];
        save([DataPath,'/Seg_frame',num2str(Ref_Frame,'%.4d')],'img_seg','Dim','Res');
        
        figure(2); imagesc(img_seg(:,:,round(end/2))); axis off; axis equal; colormap gray;
        
        
        [imgMR, meta] = nrrdread([DataPath,'/Org3D_frame',num2str(Ref_Frame),'.nrrd']);
        img = interp3(oyy, oxx, ozz, single(imgMR), yy, xx, zz);
        img(isnan(img)) = 0;
        
        nrrdWriter([DataPath,'/Img3D_frame',num2str(Ref_Frame,'%.4d'),'.nrrd'],img,[tdx, tdy, tdz],[ 0 0 0],'raw');
        

        
    end
% catch
%     if (isempty(ii) == 1)
%         disp('No available mask exists')
%     else
%         disp(['Error with mask:',name]);
%     end
% end


































