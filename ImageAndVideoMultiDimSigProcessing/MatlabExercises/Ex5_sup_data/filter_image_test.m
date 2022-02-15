% Initializing an image matrix

x = [10 10 20 20; 10 10 20 20; 10 10 30 30; 5 5 40 40];

% Initializing an 1D filter
h = 1/4 * [1 2 1];

%% Filtering using the built-in MATLAB function
y1 = imfilter(x, h)
%% Filtering using your implemented function
y2 = filter_image(x, h)
y3 = filter_image_border_handling(x, h)