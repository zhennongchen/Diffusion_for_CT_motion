function [folders] = Find_all_folders(main_path)

folder_list = dir(main_path);
dirFlags = [folder_list.isdir];
folder_list = folder_list(dirFlags);
folder_list(ismember( {folder_list.name}, {'.', '..'})) = [];
folders = folder_list;