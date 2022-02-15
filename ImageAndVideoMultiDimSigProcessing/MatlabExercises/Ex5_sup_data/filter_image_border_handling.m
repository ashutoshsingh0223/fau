function filtered_image = filter_image_border_handling(image,filter)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[M, N] = size(image);



I = size(filter, 2);

border_size = (I - 1)/2;

image_padded = padarray(image, [border_size border_size]);
[M_pad, N_pad] = size(image_padded)
filtered_image = image_padded;

% Ignoring border elements since filter is 1D on vertical borders are
% ignored
for i=1: M_pad
    for j=border_size + 1: N_pad - border_size
        sum = 0;
        for k = -border_size:border_size
            h_k = filter(k + border_size + 1);
            sum = sum + (h_k * image_padded(i, j+k));
        end
        filtered_image(i,j) = sum;
    end
end
filtered_image = filtered_image(border_size+1:end-border_size, border_size+1:end-border_size);
end

