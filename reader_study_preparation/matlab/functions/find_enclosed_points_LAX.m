function [lv_points, img_2] = find_enclosed_points_LAX(lowest_y , highest_y, p_list, Images)

lv_points = [];

for y = [lowest_y:-1:highest_y]
    disp(y)
    f = find(p_list(:,2) == y);

    if size(f,1) == 0
        p1 = [p1(1) y 1];
        p2 = [p2(1) y 1];
        length = abs(p2(1) - p1(1));
        for xx = [min([p1(1),p2(1)]) : 1 :max([p1(1),p2(1)])]
            lv_points = [lv_points; xx, y];
        end   
    end
    
    if size(f,1) == 2
        p1 = p_list(f(1),:) ;
        p2 = p_list(f(2),:);
        length = abs(p2(1) - p1(1));
        for xx = [min([p1(1),p2(1)]) : 1 :max([p1(1),p2(1)])]
            lv_points = [lv_points; xx, y];
        end    
    end
    
    if size(f,1) > 2
        xx = [];
        for jj = 1:size(f,1)
            xx = [xx p_list(f(jj), 1)];
        end
        i1 = find(xx == min(xx)); i2 = find(xx == max(xx));
            
        p1 = p_list(f(i1),:) ;
        p2 = p_list(f(i2),:);
        length = abs(p2(1) - p1(1));
        for xx = [min([p1(1),p2(1)]) : 1 :max([p1(1),p2(1)])]
            lv_points = [lv_points; xx, y];
        end    
    end

    if size(f,1) == 1  % have to interpolate
        if y == lowest_y
            p1 = p_list(f(1),:);
            p2 = p_list(f(1),:);
            length = 0;
            lv_points = [lv_points; p1(1) y];
        else

            p1_tem = p_list(f,:);

            previous_x = sort([p1(1), p2(1)]);

            previous_x_d = abs(previous_x - p1_tem(1));
            
            if length <= 5  %%% 
                length = 8
            end
            
            if previous_x_d(1) >= previous_x_d(2)
                l = length;
            else
                l = -length;
            end

            % look for the points besides it
            p2_candidate = double(p1_tem) - double([l,0,0]);
            dif = [p_list(:,1) - p2_candidate(1) , p_list(:,2) - p2_candidate(2)];
            dif = sqrt(dif(:,1) .^ 2 + dif(:,2) .^ 2);
            dif_rank = sort(dif);
            % look for two closest points
            if dif_rank(1) == dif_rank(2)
                close_index = find(dif == dif_rank(1));
            else
                close_index = [find(dif == dif_rank(1)) ; find(dif == dif_rank(2))];
            end

            close_p1 = p_list(close_index(1),:);
            close_p2 = p_list(close_index(2),:);

            close_y_list = sort([close_p1(2), close_p2(2), y]);

            if find(close_y_list == y) == 2
                disp(['two neighbours on both sides'])
                p2x = (close_p1(1) + close_p2(1)) / 2;
            elseif find(close_y_list == y) == 1
                disp(['two neighbous on apex side'])
                if close_p1(2) > close_p2(2)
                    p2x = close_p2(1) - (close_p1(1) - close_p2(1));
                else
                    p2x = close_p1(1) - (close_p2(1) - close_p1(1));
                end
            else
                disp(['two neighbous on basal side'])
                if close_p1(2) < close_p2(2)
                    p2x = close_p2(1) - (close_p1(1) - close_p2(1));
                else
                    p2x = close_p1(1) - (close_p2(1) - close_p1(1));
                end    
            end

            p2 = [p2x, y, 1];
            p1 = p1_tem;
            length = abs(p2(1) - p1(1));

            for xx = [min([p1(1),p2(1)]) : 1 :max([p1(1),p2(1)])]
                lv_points = [lv_points; xx, y];
            end  
        end

       end
end

img_2 = Images;
for ii = 1:size(lv_points,1)
    img_2(int16(lv_points(ii,1)),int16(lv_points(ii,2))) = max(Images(:))+100;
end