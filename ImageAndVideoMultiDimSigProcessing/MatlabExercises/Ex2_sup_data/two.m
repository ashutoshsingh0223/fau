shed2 = rgb2gray(imread('shed_2.png'));
shed3 = rgb2gray(imread("shed_3.png"));

figure;
imshow(shed2);
title('Shed 2');
hold on;
% 
% figure;
% imshow(shed3);
% title('Shed 3')
% 
% 
% figure;
% histogram(shed2);
% title('Shed 2 Histogram');
% 
% 
% figure;
% histogram(shed3);
% title('Shed 3 Histogram');

% figure;
% shed2_eq = histeq(shed2);
% imshow(shed2_eq);
% title('Shed 2 equalized');

%  Implement hostogram equalization

% Forming hostogram
count = zeros(1,256);
for i = 1: size(shed2,1)
    for j = 1: size(shed2, 2)
        count(shed2(i,j) + 1) = count(shed2(i,j) + 1) + 1;
    end
end

probability = count./(size(shed2,1) * size(shed2, 2));
cdf = zeros(size(probability));

for i= 1: size(probability,2)
    cdf(i) = sum(probability(1:i));
end

% Denormalizing by multiplying by 255
output_value_for_each_grey_level = uint8(round(cdf.*255));

output_pic = uint8(zeros(size(shed2)));

for i = 1: size(shed2,1)
    for j = 1: size(shed2, 2)
       output_pic(i,j) = output_value_for_each_grey_level(shed2(i,j) + 1);
    end
end

figure;
imshow(output_pic);
title('Normalized');
hold off;







