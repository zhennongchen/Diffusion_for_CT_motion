function [file_list] = Find_all_files(main_path)

file_list = dir(main_path);
file_list(ismember( {file_list.name}, {'.', '..','.DS_Store'})) = [];
