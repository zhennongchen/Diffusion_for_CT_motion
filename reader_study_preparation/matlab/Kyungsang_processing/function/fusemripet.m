function compound_RGB   = fusemripet(im2, im1, Tmap_opacity, thresholded_Tmap)
           
if nargin<3
    Tmap_opacity     = 0.1; 
    thresholded_Tmap = 0.1;
end

    temp_max_im1 = max(im1(:));
    %im1 is PET image!
    im1(im1 > thresholded_Tmap*temp_max_im1) = thresholded_Tmap*temp_max_im1;

    gray_cmap = gray(256);
    cool_cmap = jet(256);
%     cool_cmap = spring(256);

    anat_RGB   = zeros(size(im2,1), size(im2,2), 3);
    Tmap_RGB   = zeros(size(im1,1), size(im1,2), 3);

    for RGB_dim = 1:3  
        gray_cmap_rows_for_anat    = im2uint8(normalise(im2))+1;
        cool_cmap_rows_for_Tmap    = im2uint8(normalise(im1))+1;
        colour_slab_vals_for_anat  =  gray_cmap(gray_cmap_rows_for_anat, RGB_dim);
        colour_slab_vals_for_Tmap  =  cool_cmap(cool_cmap_rows_for_Tmap, RGB_dim);
        anat_RGB(:,:,RGB_dim)      = reshape(colour_slab_vals_for_anat, size(im2));
        Tmap_RGB(:,:,RGB_dim)      = reshape(colour_slab_vals_for_Tmap, size(im1));
    end
    % End of loop through the RGB dimension.

    compound_RGB = zeros(size(im2,1), size(im2,2), 3);
    %%% Loop through the three slabs: R, G, and B
    for RGB_dim = 1:3
        compound_RGB(:,:,RGB_dim) = ...
        (thresholded_Tmap==0) .* ...    % Where T-map is below threshold
        anat_RGB(:,:,RGB_dim) + ...  
        (thresholded_Tmap>0).* ...      % Where T-map is above threshold
        ( (1-Tmap_opacity) * anat_RGB(:,:,RGB_dim) + ...
        Tmap_opacity * Tmap_RGB(:,:,RGB_dim) );
        % Opacity-weighted sum of anatomical and T-map
    end
    
    compound_RGB = min(compound_RGB,1);
    
end