function [Mesh, info] = Mesh_Extraction_modified(info,Data)


info.xlim = [1500 -1500]; info.ylim = [1500 -1500]; info.zlim = [1500 -1500]; % For non-rotated LV

for j = 1:length(info.timeframes) % Time frame loop
    
    %Reading data
    if info.already_rotated == 1
        seg = Data(info.timeframes(j)).seg_rot; % rotated data
    else
        seg = Data(info.timeframes(j)).seg;
    end
    res = info.pixel_size;
    
    % xxxxxxxxxxxx IMAGE PREPARATION xxxxxxxxxxxx
    I = zeros(size(seg));
    I(seg == info.lv_label) = 1;       % Extracting LV
    I(seg == info.la_label) = 2;       % Extracting LA
    I(seg == info.lvot_label) = 4;     % Extracting LVOT
    
    % Direction matrix Transformation from .nii to .mat
    %I = permute(I,[2 1 3]);
    %I = flip(I,1);
    %I = flip(I,2);
    
    % Interpolating to make it isotropic resolution
    x = (1:size(I,2)).*res(2);
    y = (1:size(I,1)).*res(1);
    z = (1:size(I,3)).*res(3);
    xq = linspace(1*res(2),size(I,2)*res(2),round(length(x)*(res(2)/info.iso_res)));
    yq = linspace(1*res(1),size(I,1)*res(1),round(length(y)*(res(1)/info.iso_res)));
    zq = linspace(1*res(3),size(I,3)*res(3),round(length(z)*(res(3)/info.iso_res)));
    if ~all(res/info.iso_res) 
        I = interp3(x,y,z,I,xq,yq',zq,'nearest');
    end
    
    % Calculating LV volumes in ml as a function of time
    info.vol(info.timeframes(j)) = ((info.iso_res^3)*length(find(seg==1)))/1000;
    
    if info.fill_paps
        % Extracting only LV
        temp = zeros(size(I)); temp(I==1) = 1;
        
        % Cropping data set for data saving
        ind = find(temp==1);
        [row,col,zz] = ind2sub(size(temp),ind);
        
        tol = 10; %user input for cropping tolerance
        temp = temp(min(row)-tol:max(row)+tol,min(col)-tol:max(col)+tol,min(zz)-tol:max(zz)+tol);
        
        %Obtaining convex hull of LV
        s = regionprops3(temp,'BoundingBox','ConvexImage');
        
        %Resizing the bounding box
        convexhull = zeros(size(temp));
        convexhull(ceil(s.BoundingBox(2)):ceil(s.BoundingBox(2)) + s.BoundingBox(5)-1,...
            ceil(s.BoundingBox(1)):ceil(s.BoundingBox(1)) + s.BoundingBox(4)-1,...
            ceil(s.BoundingBox(3)):ceil(s.BoundingBox(3)) + s.BoundingBox(6)-1) = double(s.ConvexImage{1});
        
        %Eroding the hull to identify paps
        convexhull = imerode(convexhull,strel('sphere',8));
        
        img_subtract = convexhull - temp;
        img_subtract(img_subtract==-1) = 0;
        
        CC = bwconncomp(img_subtract,18);
        numPixels = cellfun(@numel,CC.PixelIdxList);
        
        % Identify largest component
        [~,idx]= maxk(numPixels,5);
        
        % Dilating paps to circumvent convex hull erosion
        temp2 = zeros(size(temp));
        temp2(CC.PixelIdxList{idx(1)})=1;
        temp2(CC.PixelIdxList{idx(2)})=1;
        temp2 = imdilate(temp2,strel('sphere',6));
        
        temp(temp2==1) = 1;
        
        dummy = zeros(size(I));
        dummy(min(row)-tol:max(row)+tol,min(col)-tol:max(col)+tol,min(zz)-tol:max(zz)+tol) = temp;
        
        I(dummy == 1) = 1;
        
        clear temp temp2 dummy ind s convexhull CC numPixels idx img_subtract
    end
    
    
    % Determinig angles for rotation
    if info.timeframes(j) == info.template
        if info.already_rotated == 1
            info.th_z = 0;
            info.th_x = 0;
            info.th_y = 0;
        else
            % Extracting nii image to define angles for rotation
            dummy = load_nii([info.img_path,'/img-nii-1.5/',num2str(info.timeframes(j)-1),'.nii.gz']);
            Irot = double(dummy.img);
            Irot = permute(Irot,[2 1 3]); Irot = flip(Irot,1); Irot = flip(Irot,2);
            Irot = interp3(x,y,z,Irot,xq,yq',zq);
            I_dummy = zeros(size(I)); I_dummy(I==1) = 1;
            clear dummy

            % Determining mean z-slice of LV to determine optimal angle for rotation
            [~,~,fr] = ind2sub(size(I),find(I==1));
            temp = round(mean([min(fr) max(fr)]));

            % Calculating angles - z
            figure('pos',[10 10 1000 1000])
            imagesc(Irot(:,:,temp)); hold on
            axis equal; colormap gray; caxis([-100 700])
            title('Rotate about Z axis: Click at base FIRST, THEN at apex','FontSize',30)
            [yp,zp] = ginput(2);
            info.th_z = atan(diff(yp)/diff(zp));
            close

            % z- rotation
            t_z = [cos(-info.th_z) -sin(-info.th_z) 0 0; sin(-info.th_z) cos(-info.th_z) 0 0; 0 0 1 0; 0 0 0 1];
            tform_z = affine3d(t_z);
            Irot = imwarp(Irot,tform_z);
            I_dummy = imwarp(I_dummy,tform_z,'nearest');

            % Determining mean y-slice of LV to determine optimal angle for rotation
            [~,fr,~] = ind2sub(size(I_dummy),find(I_dummy==1));
            temp = round(mean([min(fr) max(fr)]));

            % Calculating angles - x
            figure('pos',[10 10 1000 1000])
            imagesc(squeeze(Irot(:,temp,:)));
            axis equal; colormap gray; caxis([-100 700])
            title('Rotate about X axis: Click at base FIRST, THEN at apex','FontSize',30)
            [yp,zp] = ginput(2);
            info.th_x = pi/2 - atan(diff(yp)/diff(zp));
            close;
        
            % x - rotation
            t_x = [1 0 0 0; 0 cos(-info.th_x) -sin(-info.th_x) 0; 0 sin(-info.th_x) cos(-info.th_x) 0; 0 0 0 1];
            tform_x = affine3d(t_x);
            Irot = imwarp(Irot,tform_x);
            I_dummy = imwarp(I_dummy,tform_x,'nearest');

            % Determining mean z-slice of LV to determine optimal angle for rotation
            [~,~,fr] = ind2sub(size(I_dummy),find(I_dummy==1));
            temp = round(mean([min(fr) max(fr)]));

            % Calculating angles - y (rotation of LV about its long axis)
            figure('pos',[10 10 1000 1000])
            imagesc(squeeze(Irot(:,:,temp)));
            axis equal; colormap gray; caxis([-100 700])
            title('Rotate about LV axis: Click inferior wall FIRST, THEN anterior wall','FontSize',30)
            [yp,zp] = ginput(2);
            info.th_y = atan(diff(yp)/diff(zp));
            close;

            clear Irot I_dummy yp zp t_z tform_z t_x tform_x temp;
        
        end
    end
    
    clear x y z xq yq zq
    
    % xxxxxxxxxxxx MESH EXTRACTION xxxxxxxxxxxx
    im_lv = zeros(size(I)); im_lv(I==1) = 1;
    im_la = zeros(size(I)); im_la(I==2) = 1;
    im_lvot = zeros(size(I)); im_lvot(I==4) = 1;
    if info.desired_res ~= info.iso_res
        disp(['Averaging filter applied']);
        % Averaging filter
        stp = round(info.desired_res./info.iso_res.*ones(1,3));
        ff = ones(stp+1);
        ff = ff/sum(ff(:));

        %Downsampling LV
        im_lv = zeros(size(I)); im_lv(I==1) = 1;
        im_lv = imfilter(im_lv,ff,'symmetric');
        im_lv = im_lv(1:stp(1):end,1:stp(2):end,1:stp(3):end);
        im_lv = im_lv>info.averaging_threshold;

        %Downsampling LA
        im_la = zeros(size(I)); im_la(I==2) = 1;
        im_la = imfilter(im_la,ff,'symmetric');
        im_la = im_la(1:stp(1):end,1:stp(2):end,1:stp(3):end);
        im_la = im_la>info.averaging_threshold;

        %Downsampling LVOT
        im_lvot = zeros(size(I)); im_lvot(I==4) = 1;
        im_lvot = imfilter(im_lvot,ff,'symmetric');
        im_lvot = im_lvot(1:stp(1):end,1:stp(2):end,1:stp(3):end);
        im_lvot = im_lvot>info.averaging_threshold;
    end
    
    % Identify largest connected component for LV
    CC = bwconncomp(im_lv);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [~,idx]= max(numPixels);
    im_lv = zeros(size(im_lv));
    im_lv(CC.PixelIdxList{idx})=1;
    clear idx
    
    %     %Identifying largest component & downsampling LA
    %     im_la = zeros(size(I)); im_la(I==2) = 1;
    %
    %     CC = bwconncomp(im_la);
    %     numPixels = cellfun(@numel,CC.PixelIdxList);
    %     [~,idx]= max(numPixels);
    %     im_la = zeros(size(im_la));
    %     im_la(CC.PixelIdxList{idx})=1;
    %     clear idx CC numPixels
    %
    %     im_la = imfilter(im_la,ff,'symmetric');
    %     im_la = im_la(1:stp(1):end,1:stp(2):end,1:stp(3):end);
    %     im_la = im_la>info.averaging_threshold;
    %
    %     %Identifying largest component & downsampling LVOT
    %     im_lvot = zeros(size(I)); im_lvot(I==4) = 1;
    %
    %     CC = bwconncomp(im_lvot);
    %     numPixels = cellfun(@numel,CC.PixelIdxList);
    %     [~,idx]= max(numPixels);
    %     im_lvot = zeros(size(im_lvot));
    %     im_lvot(CC.PixelIdxList{idx})=1;
    %     clear idx CC numPixels
    %
    %     im_lvot = imfilter(im_lvot,ff,'symmetric');
    %     im_lvot = im_lvot(1:stp(1):end,1:stp(2):end,1:stp(3):end);
    %     im_lvot = im_lvot>info.averaging_threshold;
    
    
    %Dilating images to identify planes
    im_lv_dilate = imdilate(im_lv,ones(4,4,4));
    im_la_dilate = imdilate(im_la,ones(4,4,4));
    im_lvot_dilate = imdilate(im_lvot,ones(4,4,4));
    
    %Identifying coordinates of planes
    [ym,xm,zm] = ind2sub(size(im_lv),find((im_lv_dilate + im_la) == 2));            % mitral valve plane
    [yl,xl,zl] = ind2sub(size(im_lv),find((im_lv_dilate + im_lvot) == 2));          % lvot plane
    [yb,xb,zb] = ind2sub(size(im_lv),find((im_la_dilate + im_lvot_dilate) == 2));   % band of tissue between mitral valve and lvot
    
    info.lv_la_plane = [ym,xm,zm];
    info.lv_lvot_plane = [yl,xl,zl];
    info.la_lvot_plane = [yb,xb,zb];
    
    %Extracting mesh of LV
    [f,v] = isosurface(im_lv,0);
    %%%%% 
    % in v, the vertex coordinate is [y x z])
    %%%%%
    Mesh(info.timeframes(j)).faces = f; Mesh(info.timeframes(j)).vertices = v;
    
    %Identifying vertices that belong to planes identified previously
    [mi,~] = ismember(Mesh(info.timeframes(j)).vertices,[xm ym zm],'rows');
    [li,~] = ismember(Mesh(info.timeframes(j)).vertices,[xl yl zl],'rows');
    [bi,~] = ismember(Mesh(info.timeframes(j)).vertices,[xb yb zb],'rows');
    
    %Logical list of vertices that do not belong to planes
    ind = ~logical(mi + li + bi);
    
    % Index list of points on plane
    Mesh(info.timeframes(j)).mitral = logical(mi);
    Mesh(info.timeframes(j)).lvot = logical(li);
    Mesh(info.timeframes(j)).band = logical(bi);
    
    %Extracting faces belonging to all the vertices
    dummy = sum([ind(Mesh(info.timeframes(j)).faces(:,1)) ind(Mesh(info.timeframes(j)).faces(:,2)) ind(Mesh(info.timeframes(j)).faces(:,3))],2) == 3;
    % dummy defines whether one face is belonged to non-plane vertices
    face_objs = [Mesh(info.timeframes(j)).faces(dummy,1) Mesh(info.timeframes(j)).faces(dummy,2) Mesh(info.timeframes(j)).faces(dummy,3)];
    
    %Isolate disconnected meshes
    facecell = finddisconnsurf(face_objs);
    
    %Choosing the object with largest number of faces - LV that has the
    %boundary as those intersection planes
    num = cellfun(@numel,facecell);
    [~,idx] = max(num);
    temp = facecell{idx};
    
    %Extracting the vertices corresponding to those faces
    temp = temp(:);
    temp = unique(temp);
    
    [lia,~] = ismember(1:size(Mesh(info.timeframes(j)).vertices,1),temp);
    % indxs are the indexes of vertices that only belong to LV instead of
    % other structures & intersection planes
    Mesh(info.timeframes(j)).indxs = lia;
    
    % NaN-ing out vertices all vertices on plane or disconnected ones
    % crop_verts are all the vertices but with the ones that belong
    % structures & intersection planes as NaN.
    Mesh(info.timeframes(j)).crop_verts = NaN.*ones(size(Mesh(info.timeframes(j)).vertices));
    Mesh(info.timeframes(j)).crop_verts(temp,:) = Mesh(info.timeframes(j)).vertices(temp,:);
    
    % Identifying axes limits for mesh plotting
    if max(Mesh(info.timeframes(j)).vertices(:,1)) > info.xlim(2)
        info.xlim(2) = max(Mesh(info.timeframes(j)).vertices(:,1)) + 5*(2/info.desired_res);
    end
    
    if max(Mesh(info.timeframes(j)).vertices(:,2)) > info.ylim(2)
        info.ylim(2) = max(Mesh(info.timeframes(j)).vertices(:,2)) + 5*(2/info.desired_res);
    end
    
    if max(Mesh(info.timeframes(j)).vertices(:,3)) > info.zlim(2)
        info.zlim(2) = max(Mesh(info.timeframes(j)).vertices(:,3)) + 5*(2/info.desired_res);
    end
    
    if min(Mesh(info.timeframes(j)).vertices(:,1)) < info.xlim(1)
        info.xlim(1) = min(Mesh(info.timeframes(j)).vertices(:,1)) - 5*(2/info.desired_res);
    end
    
    if min(Mesh(info.timeframes(j)).vertices(:,2)) < info.ylim(1)
        info.ylim(1) = min(Mesh(info.timeframes(j)).vertices(:,2)) - 5*(2/info.desired_res);
    end
    
    if min(Mesh(info.timeframes(j)).vertices(:,3)) < info.zlim(1)
        info.zlim(1) = min(Mesh(info.timeframes(j)).vertices(:,3)) - 5*(2/info.desired_res);
    end
    
    % plot
    if info.timeframes(j) == 1
        figure('pos',[10 10 1200 1200]);
        patch('Faces',Mesh(info.timeframes(j)).faces,'Vertices',Mesh(info.timeframes(j)).vertices,'FaceColor','red');%,'EdgeColor','none');
        ax = gca; ax.FontSize = 20; ax.FontWeight = 'bold';
        daspect([1,1,1]); view(0,0); camlight; lighting gouraud;
        title(['Mesh for timeframe ',num2str(info.timeframes(j))],'FontSize',35); ylim(info.ylim); xlim(info.xlim); zlim(info.zlim);
        xlabel('x')
        ylabel('y')
        zlabel('z')
    end
    
    clearvars -except info Mesh j Data
    %disp(['Done with time frame ',num2str(info.timeframes(j))])
    
end

