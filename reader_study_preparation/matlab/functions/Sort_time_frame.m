function [f_sort] = Sort_time_frame(list,signal)

% list is a struct variable where the field "name" contains the file name
% signal is the symbol before the number

t = [];
for i = 1:size(list,1)
    t = [t ; Find_time_frame(list(i).name,signal)];
end

t_sort = sort(t);
f_sort = [];
for i = 1:size(list,1)
    index = find(t == t_sort(i));
    f_sort = [f_sort ; convertCharsToStrings(list(index).name)];
end