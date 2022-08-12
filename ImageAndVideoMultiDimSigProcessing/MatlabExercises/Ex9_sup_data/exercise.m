img = imread('provided/circles.png');
img_gray = rgb2gray(img);

gray_thresh = graythresh(img_gray);

gray_seg = imquantize(img_gray, gray_thresh);

histogram = hist(double(img_gray(:)), 256);

otsu_thresh = otsuthresh(histogram);

otsu_seg = imquantize(img_gray, otsu_thresh);

multi_thresh = multithresh(img, 4);

multi_seg = imquantize(img_gray, multi_thresh);

figure;

subplot(2,2,1)
imshow(uint8(gray_seg), [])
title("graythresh")

subplot(2,2,2)
imshow(uint8(otsu_seg), [])
title("otsuthresh")

subplot(2,2,3)
RGB = label2rgb(multi_seg); 
imshow(RGB)
title("multithresh")




