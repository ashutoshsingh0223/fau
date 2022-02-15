
image = imread('./provided/greenscreen.jpg');
img_red = image(:, :, 1);

histogram = hist(double(img_red(:)), 256);

top_corner = image(1:20, 1:20, :);
top_means = squeeze(mean(top_corner, [1 2]));


figure;
subplot(2,2,1)
foreground_mask = uint8(abs(img_red - top_means(1)) > 1);
red_foregound = image .* foreground_mask;
imshow(red_foregound);
title("single color threshold 1")


subplot(2,2,2)
foreground_mask = uint8(abs(img_red - top_means(1)) > 2);
red_foregound = image .* foreground_mask;
imshow(red_foregound);
title("red_green color threshold 2")

subplot(2,2,3)
foreground_mask = uint8(abs(img_red - top_means(1)) > 3);
red_foregound = image .* foreground_mask;
imshow(red_foregound);
title("single color threshold 3")

subplot(2,2,4)
foreground_mask = uint8(abs(img_red - top_means(1)) > 8);
red_foregound = image .* foreground_mask;
imshow(red_foregound);
title("single color threshold 4")


img_green = image(:, :, 3);

figure;
subplot(2,2,1)
foreground_mask_two = uint8(abs(img_red - top_means(1)) > 1 & abs(img_green - top_means(3)) > 1);
red_foregound = image .* foreground_mask_two;
imshow(red_foregound);
title("red_green color threshold 1")


subplot(2,2,2)
foreground_mask_two = uint8(abs(img_red - top_means(1)) > 2 & abs(img_green - top_means(3)) > 2);
red_foregound = image .* foreground_mask_two;
imshow(red_foregound);
title("red_green color threshold 2")

subplot(2,2,3)
foreground_mask_two = uint8(abs(img_red - top_means(1)) > 3 & abs(img_green - top_means(3)) > 3);
red_foregound = image .* foreground_mask_two;
imshow(red_foregound);
title("red_green color threshold 3")

subplot(2,2,4)
foreground_mask_two = uint8(abs(img_red - top_means(1)) > 4 & abs(img_green - top_means(3)) > 4);
red_foregound = image .* foreground_mask_two;
imshow(red_foregound);
title("red_green color threshold 4")



