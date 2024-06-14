function [Contour,point_cloud] = load_SB_contour_txt_files(txt_files)



point_cloud = [];
for t = 1:size(txt_files,2)
    file_name = txt_files(t).file_name;
    if all(file_name(1:6) == 'nofile') == 1
        Contour(t).points = [];
        continue
    end
            
    A = readtable(file_name,'ReadVariableNames',false);
   
    A = [A.Var2 A.Var1 ones(size(A.Var2)) * t];
    
    Contour(t).points = (A);
    point_cloud = [point_cloud;A];
    
end

if size(point_cloud,1) > 0
    point_cloud = sortrows(unique(point_cloud,'rows'),3);
end

