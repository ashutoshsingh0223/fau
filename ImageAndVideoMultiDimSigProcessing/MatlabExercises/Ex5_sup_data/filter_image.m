function filtered_image = filter_image(image, filter)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[M, N] = size(image);
I = size(filter, 2);
filtered_image = zeros(size(image));

border_size = (I - 1)/2;

% Ignoring border elements since filter is 1D on vertical borders are
% ignored
for i=1: M
    for j=border_size + 1: N - border_size
        sum = 0;
        for k = -border_size:border_size
            h_k = filter(k + border_size + 1);
            sum = sum + (h_k * image(i, j+k));
        end
        filtered_image(i,j) = sum;
    end
end
disp(filtered_image)
end

