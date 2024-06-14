function [I] = load_SB_png_files(png_files)

for i = 1:size(png_files,1)
    img = double(imread([png_files(i).folder,'/',png_files(i).name]));
    if i == 1
        I = zeros(size(img,1),size(img,2),size(png_files,1));
    end
    I(:,:,i) = img;
  
end