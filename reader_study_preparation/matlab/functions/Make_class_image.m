function [im_class, im_overlay, endo_inside_points_int, epi_inside_points_int] = Make_class_image(I,endo_inside_points, epi_inside_points)
% 
% [inside_points] = Enclosed_points_3D(I,Contour);

im_class = zeros(size(I));
im_overlay = I;
maxI = max(I(:));

if size(epi_inside_points,1) > 0
    epi_inside_points_int = sortrows(unique(int16(epi_inside_points),'rows'),3);
    ind = sub2ind(size(I),epi_inside_points_int(:,1),epi_inside_points_int(:,2),epi_inside_points_int(:,3));
    im_class(ind) = 2;
    im_overlay(ind) = maxI + 200;
else
    epi_inside_points_int = epi_inside_points;
end

if size(endo_inside_points,1) > 0
    endo_inside_points_int = sortrows(unique(int16(endo_inside_points),'rows'),3);
    ind = sub2ind(size(I),endo_inside_points_int(:,1),endo_inside_points_int(:,2),endo_inside_points_int(:,3));
    im_class(ind) = 1;
    im_overlay(ind) = maxI;
    
else
    endo_inside_points_int = endo_inside_points;
end



% CC = bwconncomp(im_lv);
% numPixels = cellfun(@numel,CC.PixelIdxList);
% [~,idx]= max(numPixels);
% im_lv = zeros(size(im_lv));
% im_lv(CC.PixelIdxList{idx})=1;
% clear idxs
