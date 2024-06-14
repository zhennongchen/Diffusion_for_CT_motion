function [X]=Name_new_file(qianzhui,number,houzhui,total_length)
%the final name should be like Image_0011.dcm
n = num2str(number);
l = length(n);
add = ['000000000'];
X = [add(1:(total_length-l)),n];
X = [qianzhui,X,houzhui];