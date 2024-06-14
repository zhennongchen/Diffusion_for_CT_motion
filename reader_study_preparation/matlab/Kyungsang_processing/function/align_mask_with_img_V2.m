function out = align_mask_with_img_V2(mesh,dx,dy,dz)
 
	out = mesh;
    out.points(:,1:2) = -mesh.points(:,1:2);
    out.points(:,1) = mesh.points(:,1)./dx;
    out.points(:,2) = mesh.points(:,2)./dy;
    out.points(:,3) = mesh.points(:,3)./dz;

end