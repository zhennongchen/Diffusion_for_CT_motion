function [J] = Turn_data_into_greyscale(image,window_level,window_width)

siz = size(image);
if length(siz) == 3
    J = ones(siz);
    
    for i = 1:siz(length(siz))
        J(:,:,i) = mat2gray(image(:,:,i),[window_level-window_width,window_level+window_width]);
    end
else
    J = mat2gray(image,[window_level-window_width,window_level+window_width]);
end