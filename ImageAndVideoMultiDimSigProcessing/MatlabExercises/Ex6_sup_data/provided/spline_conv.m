function [x_n,y_n] = spline_conv(y_n_1, y0)
%SPLINE_CONV Summary of this function goes here
%   Detailed explanation goes here
n_1 = size(y_n_1, 2);
n0 = size(y0, 2);

y_n = zeros(1, n_1 + n0 - 1);
n = size(y_n,2);


padding = (n - n0)/2;
pad_filter = padarray(y0, [0  padding], "both");
% Since y_0 is symmetric no need to filp the kernel
for shift=-(n - 1)/2: (n - 1)/2
    sum = 0;
    shifted = circshift(pad_filter, shift , 2);
    for k=-(n_1 - 1)/2: (n_1 - 1)/2
        index = k + (n_1 - 1)/2 + 1;
        sum = sum + y_n_1(index) * shifted(index); 
    end
    y_n(shift + (n - 1)/2 + 1) = sum;
end
[~, argmin] = max(y_n);
x_n = -(n - 1)/2 + argmin: (n - 1)/2 + argmin;

end


