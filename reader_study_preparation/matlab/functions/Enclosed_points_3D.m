function [inside_points] = Enclosed_points_3D(I,IContour)

inside_points = [];

if size(size(I),2) == 2
    dim = 2; % 2D
    series = 1;
elseif size(size(I),2) == 3
    dim = 3; % 3D
    series = [1:size(I,3)];
end

for t = series
   
    P = double(IContour(t).points);
    if size(P,1) == 0
        continue
    end
        
    X = P(:,1);Y = P(:,2);

    k = boundary(X,Y,0);
    inside_points = [inside_points;P];
    
    [meshx,meshy] = meshgrid(1:256,1:256);
    indices = inpolygon(reshape(meshx,1,[]),reshape(meshy,1,[]),X(k),Y(k));
    indices = reshape(indices,size(I,1),size(I,2));
    [ii,jj] = ind2sub([size(I,1),size(I,2)],find(indices==1));
    
    if dim == 3
        add_points = [jj,ii,ones(size(ii)) * t];
        inside_points = [inside_points; add_points];
        inside_points = sortrows(unique(inside_points,'rows'),3);
    elseif dim == 2
        add_points = [jj,ii,ones(size(ii)) * t];
        inside_points = [inside_points; add_points];
        inside_points = sortrows(unique(inside_points,'rows'),1);
    end
        
    
    if t == 1
        figure()
        plot(X,Y,'b.');
        hold on
        plot(X(k),Y(k),'k.')
        if dim == 3
            aa = find(inside_points(:,3) == t);
            inside_points_slice = inside_points(aa,:);
        elseif dim == 2
            inside_points_slice = inside_points;
        end
  
        plot(inside_points_slice(:,1),inside_points_slice(:,2),'r.')
    end
end