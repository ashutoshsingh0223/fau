img = imread('provided/colors.png');

random_red = randi([min(img(:,:,1), [], "all") max(img(:,:,1), [], "all")], 1, 4);

random_green = randi([min(img(:,:,2), [],  "all") max(img(:,:,2), [], "all")], 1, 4);

result = zeros(size(img,1), size(img,2));

% Calculating new centroids here itself

new_red = zeros(1, 4);
new_green = zeros(1, 4);
counts = zeros(1, 4);

for i=1: size(img,1)
    for j=1: size(img, 2)
        p_red = img(i,j, 1);
        p_green = img(i,j, 2);
        red_dist = random_red - double(p_red);
        green_dist = random_green - double(p_green);
        [~, min_cluster_idx] = min(sqrt(red_dist.^2 + green_dist.^2));
        result(i, j) = min_cluster_idx;
%        Collect data for new centroids
        new_red(min_cluster_idx) = new_red(min_cluster_idx) + double(p_red);
        new_green(min_cluster_idx) = new_green(min_cluster_idx) + double(p_green);
        counts(min_cluster_idx) = counts(min_cluster_idx) + 1;
    end
end

seg_mask = label2rgb(result);

figure;
subplot(1,2, 1)
imshow(uint8(seg_mask));
title("with label2rgb")

subplot(1,2, 2)
imagesc(result)
title("with imagesc")



new_red = new_red ./ counts;
new_green = new_green ./ counts;

% Store centroids for 100 iterations
centroids = zeros(100, 4, 2);
new_result = zeros(size(img,1), size(img,2));

for iter=1:100
    new_red = new_red ./ counts;
    new_green = new_green ./ counts;
    updated_red = zeros(1, 4);
    updated_green = zeros(1, 4);
    updated_counts = zeros(1, 4);
    centroids(iter,:,1) = new_red;
    centroids(iter,:,2) = new_green;
    for i=1: size(img,1)
        for j=1: size(img, 2)
            p_red = img(i,j, 1);
            p_green = img(i,j, 2);
            red_dist = new_red - double(p_red);
            green_dist = new_green - double(p_green);
            [~, min_cluster_idx] = min(sqrt(red_dist.^2 + green_dist.^2));
            new_result(i, j) = min_cluster_idx;
            % Collect data for new centroids
            updated_red(min_cluster_idx) = updated_red(min_cluster_idx) + double(p_red);
            updated_green(min_cluster_idx) = updated_green(min_cluster_idx) + double(p_green);
            updated_counts(min_cluster_idx) = updated_counts(min_cluster_idx) + 1;
        end
    end
    new_red = updated_red;
    new_green = updated_green;
    counts = updated_counts;
end


new_seg_mask = label2rgb(new_result);

figure;
subplot(1,2, 1)
imshow(uint8(new_seg_mask));
title("with label2rgb new segmask")

subplot(1,2, 2)
imagesc(new_result)
title("with imagesc new segmask")


% figure;
% subplot(2,2,1)
% hond on;
% for i=1:100
%    plot(centroids)
% end



