function output = manualsplit(cur_line, spli)

    idx = (cur_line == spli);
    item_sum = sum(idx)+1;
    output = cell(1,item_sum);
    
    st_idx = [1, find(cur_line == spli)+1];

    ed_idx = [find(cur_line == spli)-1, length(cur_line)];

    for t = 1:length(st_idx)
        if(~(st_idx(t) == ed_idx(t)))
            output{1,t} = cur_line(st_idx(t):ed_idx(t));
        end
    end
end