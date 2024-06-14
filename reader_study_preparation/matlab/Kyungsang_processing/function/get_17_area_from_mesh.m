function [class_endo, class_epi, class_aod1, class_aod2] = get_17_area_from_mesh(varargin)
%%
% Didive the point cloud into 17 areas
% Input:
% 3D coordinate of endo (N1 X 3)
% 3D coordinate of epi  (N2 X 3)
% 3D coordinate of displacement field of endo(option)
% 3D coordinate of displacement field of epi(option)
% 
% Output:
% Class of endo range from 1 to 17 (N1 X 1)
% Class of epi range from 1 to 17 (N2 X 1)
% Class of endo displacement field range from 1 to 17(option)
% Class of epi displacement field range from 1 to 17 (option)
%
% Author:
% Zhiling Zhou
% Date:
% 12/11/2018
%%
    endo = varargin{1};
    epi = varargin{2};
       
    if nargin == 2
        aod1 = endo;
        aod2 = epi;
    elseif nargin ==4
        aod1 = varargin{3};
        aod2 = varargin{4};
    end
    
    size_endo = length(endo);
    size_epi = length(epi);
    size_aod1 = length(aod1);
    size_aod2 = length(aod2);
        
    % move central to (0 0 0)
    %central = [0 0 0]; 
    centralepi = mean(epi);
    epi = epi - repmat(centralepi,length(epi),1);
    aod2(:,1:3) = aod2(:,1:3) - repmat(centralepi,length(aod2),1);
    
%    centralendo = mean(endo);
    endo = endo - repmat(centralepi,length(endo),1);
    aod1(:,1:3) = aod1(:,1:3) - repmat(centralepi, length(aod1),1);

    xmin = min(epi(:,1)); xmax = max(epi(:,1));
    ymin = min(epi(:,2)); ymax = max(epi(:,2));
    zmin = min(epi(:,3)); zmax = max(epi(:,3));
    xymin = min(xmin,ymin); xymax = max(xmax,ymax);

    % cut horizontally
    basal_height = max(epi(:,3));
    bottom_height = min(endo(:,3));
    rest_height = basal_height - bottom_height;
    each_part_height = rest_height / 3;
    top_height = basal_height - each_part_height;
    middle_height = basal_height - 2*each_part_height;
    % Analytic:
    % z = middle_height
    % z = top_height
    % z = bottom_height
        
    % clear allepi allendo;
    % divide the vertices in height
%    pc = [epi;endo];
    pc_withaod = [endo; epi; aod1(:,1:3); aod2(:,1:3)];
    
    class = zeros(size(pc_withaod,1),1);
    class(pc_withaod(:,3) < bottom_height) = 17;
    class(pc_withaod(:,3) > top_height) = -1;
    idx1 = find(pc_withaod(:,3) > middle_height);
    idx2 = find(pc_withaod(:,3) <= top_height);
    idx3 = find(pc_withaod(:,3) >= bottom_height);
    idx4 = find(pc_withaod(:,3) <= middle_height);
    class(intersect(idx1,idx2)) = -2;
    class(intersect(idx3,idx4)) = -3;
        
    % Apical divide
        class((pc_withaod(:,1) <= pc_withaod(:,2))&...
                     (pc_withaod(:,1) < -pc_withaod(:,2))&...
                     (class==-3))        = 14;
        class((pc_withaod(:,1) > -pc_withaod(:,2))&...
                     (pc_withaod(:,1) <= pc_withaod(:,2))&...
                     (class==-3))        = 13;          
        class((pc_withaod(:,1) >= pc_withaod(:,2))&...
                     (pc_withaod(:,1) > -pc_withaod(:,2))&...
                     (class==-3))        = 16;    
        class((pc_withaod(:,1) < -pc_withaod(:,2))&...
                     (pc_withaod(:,1) >= pc_withaod(:,2))&...
                     (class==-3))        = 15;
                 
    % Mid divide
        class((pc_withaod(:,2) > 0)&...
                     (pc_withaod(:,1) >= pc_withaod(:,2)/3^(1/2))&...
                     (class==-2))        = 12;
        class((pc_withaod(:,2) <= 0)&...
                     (pc_withaod(:,1) > -pc_withaod(:,2)/3^(1/2))&...
                     (class==-2))        = 11;
        class((pc_withaod(:,2) < 0)&...
                     (pc_withaod(:,1) <= pc_withaod(:,2)/3^(1/2))&...
                     (class==-2))        = 9;
        class((pc_withaod(:,2) >= 0)&...
                     (pc_withaod(:,1) < -pc_withaod(:,2)/3^(1/2))&...
                     (class==-2))        = 8;
        class((pc_withaod(:,1) >= -pc_withaod(:,2)/3^(1/2))&...
                     (pc_withaod(:,1) < pc_withaod(:,2)/3^(1/2))&...
                     (class==-2))        = 7;
        class((pc_withaod(:,1) > pc_withaod(:,2)/3^(1/2))&...
                     (pc_withaod(:,1) <= -pc_withaod(:,2)/3^(1/2))&...
                     (class==-2))        = 10;
                 
        % Basal divide
        class((pc_withaod(:,2) > 0)&...
                     (pc_withaod(:,1) >= pc_withaod(:,2)/3^(1/2))&...
                     (class==-1))        = 6;
        class((pc_withaod(:,2) <= 0)&...
                     (pc_withaod(:,1) > -pc_withaod(:,2)/3^(1/2))&...
                     (class==-1))        = 5;
        class((pc_withaod(:,2) < 0)&...
                     (pc_withaod(:,1) <= pc_withaod(:,2)/3^(1/2))&...
                     (class==-1))        = 3;
        class((pc_withaod(:,2) >= 0)&...
                     (pc_withaod(:,1) < -pc_withaod(:,2)/3^(1/2))&...
                     (class==-1))        = 2;
        class((pc_withaod(:,1) >= -pc_withaod(:,2)/3^(1/2))&...
                     (pc_withaod(:,1) < pc_withaod(:,2)/3^(1/2))&...
                     (class==-1))        = 1;
        class((pc_withaod(:,1) > pc_withaod(:,2)/3^(1/2))&...
                     (pc_withaod(:,1) <= -pc_withaod(:,2)/3^(1/2))&...
                     (class==-1))        = 4;
        
    pc_withaod = [endo; epi; aod1(:,1:3); aod2(:,1:3)];
    
    
    class_endo = class(1:size_endo);
    class_epi =  class(size_endo+1:size_endo+size_epi);
    class_aod1 = class(size_endo+size_epi+1:size_endo+size_epi+size_aod1);
    class_aod2 = class(end-size_aod2+1:end);
    
  
end