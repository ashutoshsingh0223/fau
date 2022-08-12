function [ssd] = calc_ssd(image,refer_image)
%CALC_SSD Summary of this function goes here
%   Detailed explanation goes here

ssd = sum(sum((image - refer_image) .^ 2));
end

