function output = delete_useless_pt(SAX_mesh)

    faces = SAX_mesh.faces;
    vertices = SAX_mesh.vertices;
    
    pc_seq = linspace(1,length(SAX_mesh.vertices),length(SAX_mesh.vertices));
    pc = unique(pc_seq);
    
    idx = ismember(faces,pc);
    face_new = faces(sum(idx,2)==3,:);
    
    
    used_vertices_seq = unique(face_new);
    used_vertices_num = length(used_vertices_seq);
    vertices_new = vertices(used_vertices_seq,:);
    
    correspond = [used_vertices_seq'];%;linspace(1,used_vertices_num,used_vertices_num)]
    
    for t = 1:size(correspond,2)
        %oldnum = correspond(t);
        face_new(face_new==correspond(t)) = t;
    end    
   
    output.faces = face_new;
    output.vertices = vertices_new;

end