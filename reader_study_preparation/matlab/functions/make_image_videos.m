function make_image_videos(video_name,I)

figure_size = [10 10 300 300];

writerObj = VideoWriter(video_name,'MPEG-4');
writeObj.Quality = 100;
writerObj.FrameRate = 5;
    
% open the video writer
open(writerObj);

maxI = max(I(:));
% write the frames to the video
for t = 1:size(I,3)
    close all;
    img_label = I(:,:,t);
    
    h = figure('pos',figure_size);
    imagesc(img_label);
    axis off
    frame = getframe(h);
    writeVideo(writerObj, getframe(gcf));
    close all
end

close(writerObj);