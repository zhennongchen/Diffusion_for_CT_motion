function [t] = Find_time_frame(file_name,signal)

dot_pos = strfind(file_name,'.');
if signal ~= 'no'
    signal_pos = strfind(file_name,signal);
    t = file_name(signal_pos(end)+1 : dot_pos(1)-1);
else
    signal_pos = 1;
    t = file_name(1 : dot_pos(1)-1);
end
t = str2num(t);



