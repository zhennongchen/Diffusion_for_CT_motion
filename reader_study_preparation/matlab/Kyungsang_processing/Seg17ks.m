function [img_seg] = Seg17ks(img_outer,img_inner,img_mark,res)
%SEG17_FUNCTION Summary of this function goes here
%   Detailed explanation goes here

[tnx, tny, tnz] = size(img_outer);
tdx = res(1); tdy = res(2); tdz = res(3);


x = single([0:tnx-1] - (tnx-1.0)/2.0)*tdx;
y = single([0:tny-1] - (tny-1.0)/2.0)*tdy;
z = single([0:tnz-1] - (tnz-1.0)/2.0)*tdz;

[yy, xx, zz] = meshgrid(y, x, z);

L = bwlabeln(img_mark);
for im = 1:2
    id = find(L==im);
    id_mark(im) = id(1);
end

LM1 = [ xx(id_mark(1)), yy(id_mark(1)), yy(id_mark(1))];
LM1_id = (LM1./[tdx, tdy, tdz] + [(tnx-1.0)/2.0, (tny-1.0)/2.0, (tnz-1.0)/2.0])+1;
LM2 = [ xx(id_mark(2)), yy(id_mark(2)), zz(id_mark(2))];

tmp = img_outer(:,:,LM1_id(3));
CM_id = [ sum(sum(tmp,2)'.*[1:tnx])/sum(tmp(:)), sum(sum(tmp,1).*[1:tny])/sum(tmp(:))];
CM = (CM_id-1- [(tnx-1.0)/2.0,(tny-1.0)/2.0]).*[tdx,tdy];


if mod((atan2(LM1(1)-CM(1),LM1(2)-CM(2))-atan2(LM2(1)-CM(1),LM2(2)-CM(2)))/pi*180,360)>180
    tmp = LM1;
    LM1 = LM2;
    LM2 = tmp;
end

LM3= (LM1+LM2)/2;



zzz = squeeze(sum(sum(img_inner,1),2));

z_max_id = max(find(zzz>0));
z_min_id = min(find(zzz>0));
z_max = (z_max_id-1- (tnz-1.0)/2.0).*[tdz];
z_min = (z_min_id-1- (tnz-1.0)/2.0).*[tdz];

id = find(max(img_outer-img_inner,0)>0);
img_seg = zeros(tnx, tny, tnz, 'single');

theta1 = mod(atan2(LM1(1)-CM(1),LM1(2)-CM(2))/pi*180,360);
theta2 = mod(atan2(LM2(1)-CM(1),LM2(2)-CM(2))/pi*180,360);
theta3 = mod(atan2(LM3(1)-CM(1),LM3(2)-CM(2))/pi*180,360);

theta1n = mod(theta1+180,360);
theta2n = mod(theta2+180,360);
theta3n = mod(theta3+180,360);

cen6 = [(theta1+theta2n)/2, (theta1+theta3)/2, (theta2+theta3)/2, (theta2+theta1n)/2, (theta1n+theta3n)/2, (theta2n+theta3n)/2];
if abs(theta1-theta2n)>180 cen6(1) = cen6(1)+180; end
if abs(theta1-theta3)>180 cen6(2) = cen6(2)+180; end
if abs(theta2-theta3)>180 cen6(3) = cen6(3)+180; end
if abs(theta2-theta1n)>180 cen6(4) = cen6(4)+180; end
if abs(theta1n-theta3n)>180 cen6(5) = cen6(5)+180; end
if abs(theta2n-theta3n)>180 cen6(6) = cen6(6)+180; end
cen6 = mod(cen6,360);


cen4 = mod([theta3+90, theta3, theta3-90, theta3+180],360);



for iid = 1:length(id)
    pos = [xx(id(iid)), yy(id(iid)), zz(id(iid))];
    
    theta = mod(atan2(pos(1)-CM(1),pos(2)-CM(2))/pi*180,360);
    
    if pos(3)<z_min
        img_seg(id(iid)) = 17;
    elseif pos(3)>=z_min && pos(3)<z_min+1/3*(z_max-z_min) % seg 13~16
        
        angle_diff = abs(theta - cen4);
        for j=1:4
            if angle_diff(j)>180
                angle_diff(j) = abs(angle_diff(j)-360);
            end
        end
        
        seg = find(angle_diff == min(angle_diff));
        seg = seg(1)+12;
        
        
        img_seg(id(iid)) = seg;
        
    elseif pos(3)>=z_min+1/3*(z_max-z_min) && pos(3)<z_min+2/3*(z_max-z_min) % seg 7~12
        
        angle_diff = abs(theta - cen6);
        for j=1:6
            if angle_diff(j)>180
                angle_diff(j) = abs(angle_diff(j)-360);
            end
        end
        
        seg = find(angle_diff == min(angle_diff));
        seg = seg(1)+6;
        
        img_seg(id(iid)) = seg;
        
    else % seg 1~6
        angle_diff = abs(theta - cen6);
        for j=1:6
            if angle_diff(j)>180
                angle_diff(j) = abs(angle_diff(j)-360);
            end
        end
        
        seg = find(angle_diff == min(angle_diff));
        seg = seg(1);
        
        img_seg(id(iid)) = seg;
        
    end
end


end

