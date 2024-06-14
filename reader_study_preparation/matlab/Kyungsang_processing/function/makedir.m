function res = makedir(root)
    if(~exist(root,'dir')) 
        res = mkdir(root); 
    end
end


