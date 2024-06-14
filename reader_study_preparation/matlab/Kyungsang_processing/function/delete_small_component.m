function [out_pc, mesh, mask] = delete_small_component(pc, alpha, sz)

    disp('Deleting Epi Small Component...')
    pc_epi = pc(pc(:,4)>0,:);
    pc_endo = pc(pc(:,4)<0,:);
    out_pc = [];
    mask = zeros(sz);
    
    for k = 1:2
        if(k==1)
            cur_pc = pc_epi;
            label = 1;
        else
            cur_pc = pc_endo;
            label = -1;
        end
        shp = alphaShape(cur_pc(:,1),cur_pc(:,2),cur_pc(:,3));
        shp.Alpha = alpha;
        [mesh_st.faces, mesh_st.vertices] = alphaTriangulation(shp,1);
        mesh_st.faces = mesh_st.faces(:,1:3);
        mesh = delete_useless_pt(mesh_st);

        tmp = mesh.vertices;
        out_pc = [out_pc; tmp, ones(length(tmp),1)*label];
        for i = 1:length(tmp)
            mask(tmp(i,1),tmp(i,2),tmp(i,3)) = label*255;
        end
    end
    disp('Deleting Small Component done!')
end