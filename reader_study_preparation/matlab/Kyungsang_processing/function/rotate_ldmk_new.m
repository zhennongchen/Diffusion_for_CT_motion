function [output, LMS, LME] = rotate_ldmk_new(pc,LMS,LME,DIRE)
%%
% Rotate pointcloud according to two landmarks and one direction
% Input:
% pc: pointcloud
% LMS: landmark1
% LMS: landmark2
% DIRE: direction point
% 
% Output:
% output: rotated pointcloud
% LMS: rotated landmark1
% LME: rotated landmark2
%
% Author:
% Zhiling Zhou
% Date:
% 12/18/2018
%%
    ROT = LMS - LME;
    X = ROT(1);
    Y = ROT(2);
    Z = ROT(3);

    
    %Psi = acos(Y/(sqrt(X^2 + Y^2)))*180/pi;
    Psi = acos(abs(Y)/(sqrt(X^2 + Y^2)))*180/pi;
    
    coef1 = LMS(1)-LME(1);
    coef2 = LMS(2)-LME(2);
    
    if(coef1<0)
        coef1 = -coef1;
        coef2 = -coef2;
    end
    
    res = DIRE(2)*coef1 - DIRE(1)*coef2 - (LMS(2)*coef1 - LMS(1)*coef2);
    
     if(res > 0)
         Psi = 180-Psi ;
     end

    %Psi = 180;
    RzNPsi = [  cosd(Psi)  -sind(Psi) 0;
                sind(Psi)  cosd(Psi)  0;
                0           0           1];

    pointNum = length(pc(:,1));
    center = (max(pc(:,1:3)) + min(pc(:,1:3)))/2;
    origin = repmat(center, pointNum,1);

    if(size(pc,2)>3)
        output = [(pc(:,1:3) - origin) * RzNPsi + origin, pc(:,4:end)];
    else
        output = (pc(:,1:3) - origin) * RzNPsi + origin;
    end
    LMS = (LMS - center) * RzNPsi + center;
    LME = (LME - center) * RzNPsi + center;
end