image = imread('shapes.png');
red = image(:,:,1);
green = image(:,:,2);
blue = image(:,:,3);

% This plots the image as greyscale
imshow(red);
imshow(green);
imshow(blue);

zero = zeros(size(red));

figure;
only_red = cat(3, red, zero, zero);
imshow(only_red);

figure;
only_green = cat(3, zero, green, zero);
imshow(only_green);

figure;
only_blue = cat(3, zero, zero, blue);
imshow(only_blue);