function make_contour_videos(video_name,I,endo_contour, epi_contour)

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
    endo_contour_p = int16(endo_contour(t).points);
    for i = 1:size(endo_contour_p,1)
        img_label(endo_contour_p(i,1),endo_contour_p(i,2)) = maxI ;
    end
    
    epi_contour_p = int16(epi_contour(t).points);
    for i = 1:size(epi_contour_p,1)
        img_label(epi_contour_p(i,1),epi_contour_p(i,2)) = maxI + 200;
    end
    
    h = figure('pos',figure_size);
    imagesc(img_label);
    axis off
    frame = getframe(h);
    writeVideo(writerObj, getframe(gcf));
    close all
end

close(writerObj);