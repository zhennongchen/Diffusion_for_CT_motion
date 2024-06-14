function out = to_plot_point_cloud(res_plot, threshhold,if_delete_zero)

size_res = size(res_plot);
len = size_res(1);

A = 1:size_res(1);
B = 1:size_res(2);
C = 1:size_res(3);

[a,b,c] = ndgrid(A, B, C);
sca = [a(:) b(:) c(:)];

ptsize = zeros(size(sca,1),1);
ptsizetop = zeros(size(sca,1),1);

for a = 1:size(res_plot,1)
    for b = 1:size(res_plot,2)
      ptsize(1+len*(b-1)+len*len*(a-1):len*b+len*len*(a-1)) = res_plot(:,b,a);
    end
end

if(if_delete_zero)
    sca_nozero = sca(ptsize > threshhold & ptsize ~= 0,:);
    size_nozero = ptsize(ptsize > threshhold & ptsize ~= 0);
else
    sca_nozero = sca(ptsize > threshhold,:);
    size_nozero = ptsize(ptsize > threshhold);
end
out = [sca_nozero, size_nozero];


end