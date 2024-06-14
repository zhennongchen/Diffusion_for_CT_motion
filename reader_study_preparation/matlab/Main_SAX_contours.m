%% Preparation: Data Organization
clear all; close all; clc;
code_path = '/Users/zhennongchen/Documents/GitHub/CMR_HFpEF_Analysis/matlab';
addpath(genpath(code_path));
main_path = '/Users/zhennongchen/Documents/Zhennong_CMR_HFpEF/Data/Sunny_Brooks/';
data_path = '/Users/zhennongchen/Documents/Zhennong_CMR_HFpEF/Data/Sunny_Brooks/CMR_SAX_SB';
LAX_path = '/Users/zhennongchen/Documents/Zhennong_CMR_HFpEF/Data/Sunny_Brooks/CMR_LAX_SB';
save_path = '/Users/zhennongchen/Documents/Zhennong_CMR_HFpEF/Data/Sunny_Brooks/Processed_Data';
%% SAX
cases = Find_all_folders(data_path);
for c = 2:size(cases,1)
    case_id = cases(c).name;
    disp(case_id)
    data_folder = [data_path,'/',case_id];
    make_video = 1;
    save_folder = [save_path,'/',case_id,'/SAX'];
    mkdir([save_path,'/',case_id])
    mkdir(save_folder)
    
    % find two time frames and their particular file name
    png_files = Find_all_files([data_folder,'/png/','*.png']);
    time_frame_signal = [];
    
    for i = 1:size(png_files,1)
        file_name = png_files(i).name;
        signal_pos = strfind(file_name,'.');
        signal = file_name(signal_pos-1);
        time_frame_signal = [time_frame_signal, str2num(signal)];
        if i == 1
            tf1 = str2num(signal);
        end
    end
    time_frame_signal = unique(time_frame_signal);

    if size(time_frame_signal,2) < 2
        disp('less than two time frames!!!')
        tf1 = time_frame_signal(1);
        tf2 = time_frame_signal(1);

    elseif size(time_frame_signal ,2) == 2
        tf2 = time_frame_signal(find(time_frame_signal ~= tf1));
        
    else
        error('more than two time frames!!!')
    end

    clear signal signal_pos file_name png_files

% make contour points for each time frame
    for i = [1,2] %size(time_frame_signal,2)
        clear endo_contour endo_point_cloud epi_contour epi_point_cloud endo_txt_files epi_txt_files video_name endo_inside_points epi_inside_points im_class im_overlay

        if size(time_frame_signal,2) == 2

            time_frame = time_frame_signal(i);

            if time_frame == tf1
                tf = 'tf1';
            else
                tf = 'tf2';
            end

            % load png files
            keywords_png = ['*',num2str(time_frame),'.png'];
            png_files = Find_all_files([data_folder,'/png/',keywords_png]);
        end

        if size(time_frame_signal,2) == 1
            time_frame = time_frame_signal(1);
            png_files = Find_all_files([data_folder,'/png/',['*png']]);
            if i == 1
                tf = 'tf1';
                png_files = png_files([1:2:size(png_files,1)],:);
            else
                tf = 'tf2';
                png_files = png_files([2:2:size(png_files,1)],:);
            end

        end

        [Images] = load_SB_png_files(png_files);

        % load contour points file
        % first do endo-contours
        for j = 1:size(png_files,1)
            case_name = png_files(j).name;
            signal_pos = strfind(case_name,'.');
            case_name = case_name(1:signal_pos-1);

            txt_file = Find_all_files([data_folder,'/txt/',[case_name, '-icon*']]);
            if size(txt_file,1) > 0
                file_name = [txt_file.folder,'/',txt_file.name];
                endo_txt_files(j).file_name = file_name;
            elseif size(txt_file,1) == 0
                endo_txt_files(j).file_name = 'nofile';
            end
        end
        [endo_contour,endo_point_cloud] = load_SB_contour_txt_files(endo_txt_files);

        % second do epi-contours
        for j = 1:size(png_files,1)
            case_name = png_files(j).name;
            signal_pos = strfind(case_name,'.');
            case_name = case_name(1:signal_pos-1);

            txt_file = Find_all_files([data_folder,'/txt/',[case_name, '-ocon*']]);
            if size(txt_file,1) == 1
                file_name = [txt_file.folder,'/',txt_file.name];
                epi_txt_files(j).file_name = file_name;
            elseif size(txt_file,1) == 0
                epi_txt_files(j).file_name = 'nofile';
            end
        end
        [epi_contour,epi_point_cloud] = load_SB_contour_txt_files(epi_txt_files);


        clear case_name signal_pos txt_file file_name j

        % make contour videos
        if make_video == 1
            mkdir([save_folder,'/Contour_videos'])
            video_name = [save_folder,'/Contour_videos/',case_id,'_',tf,'_signal',num2str(time_frame),'.mp4'];
            make_contour_videos(video_name,Images, endo_contour, epi_contour);
        end


        % find LV blood pool and myocardium
        [endo_inside_points] = Enclosed_points_3D(Images,endo_contour);
        [epi_inside_points] = Enclosed_points_3D(Images, epi_contour);
        if (size(endo_inside_points,1) > 0) && (size(epi_inside_points,1) > 0)
            [~,~,intersect_index_epi] = intersect(endo_inside_points, epi_inside_points,'rows');
            epi_inside_points(intersect_index_epi,:) = [];
        end
        close all

        % make categorical image
        [im_class, im_overlay, endo_inside_points_int, epi_inside_points_int] = Make_class_image(Images,endo_inside_points, epi_inside_points);
        figure()
        imagesc(im_class(:,:,3))
        
        if make_video == 1
            mkdir([save_folder,'/Contour_videos'])
            video_name = [save_folder,'/Contour_videos/',case_id,'_',tf,'_signal',num2str(time_frame),'_img.mp4'];
            make_image_videos(video_name,im_overlay);
        end

        % save
        save_file = [save_folder,'/',case_id,'_',tf,'.mat'];
        save(save_file, 'Images', 'im_class', 'case_id','time_frame_signal','tf1', 'tf2', 'endo_inside_points', 'endo_inside_points_int', 'epi_inside_points', 'epi_inside_points_int')
    end
end

