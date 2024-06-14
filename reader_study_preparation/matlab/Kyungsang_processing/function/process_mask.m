function [pc, mesh, mask] = process_mask(mesh,sz,smooth_iteration)
    
    FV.vertices = mesh.points;
    FV.faces = mesh.triangles;
    
    disp('Smoothing...')
    while(smooth_iteration>0)
        % FV = smoothpatch(st,mode,iteration);
        FV.vertices = lpflow_trismooth(FV.vertices,FV.faces);
        smooth_iteration = smooth_iteration - 1;
    end
    disp('Smooth done!')
    
    center = mean(FV.vertices);
    dist = FV.vertices - center;
    n_pt = FV.vertices - 0.01 * dist;
    in_out = inpolyhedron(FV,n_pt);

    in_out = in_out .* 2 - 1;

    FV.vertices = [FV.vertices, in_out];
    vertices = [FV.vertices, in_out];
    vertices(:,1:3) = round(vertices(:,1:3))+1;
    vertices = unique(vertices,'rows');
    
    mask = zeros(sz);
    for kk = 1:length(vertices)
        mask(vertices(kk,2),vertices(kk,1),vertices(kk,3)) = 255 .* vertices(kk,4);
    end
    
    threshhold = -1e5;
    pc = to_plot_point_cloud(mask, threshhold, 1);

    % delete small components
    alpha = 5;
    [pc, mesh, mask] = delete_small_component(pc, alpha, sz);
end