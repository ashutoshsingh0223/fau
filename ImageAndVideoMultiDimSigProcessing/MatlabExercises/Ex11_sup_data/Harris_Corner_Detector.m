clear all
close all

%% Load image
img = checkerboard(16);
img = (img > 0.5);
img = im2double(img);

%% Initalize parameters
w = fspecial('gaussian',[3 3],2);
k = 0.01;
height = size(img,1);
width = size(img,2);

%% Compute local derivatives of image
% s_x(1, :) = zeros(1, size(img,2));
for x = 2:size(img,1)
    s_x(x,:) = img(x, :) - img(x-1, :);
end

% s_y(:, 1) = zeros(size(img,1), 1);
for y = 2:size(img,2)
    s_y(:,y) = img(:, y) - img(:, y-1);
end

%% Compute products of local derivatives at every pixel
s_xx = s_x .* s_x;
s_xy = s_x  .* s_y;
s_yy = s_y .* s_y;

%% Compute sums of weighted products of derivatives at each pixel

s_xx = filter2(w, s_xx, "same");
s_xy = filter2(w, s_xy, "same");
s_yy = filter2(w, s_yy, "same");

%% Define at each pixel (x,y) Harris structure matrix M
for i = 1:height
    for j = 1:width
        M = [s_xx(i,j) s_xy(i,j); s_xy(i,j) s_yy(i,j)];
            
        % Compute cornerness c[x,y] at each pixel (x,y)
        c(i,j) = det(M) - k*(trace(M) ^ 2);
    end
end

%% Apply nonmax suppression

% Define threshold
c_max = max(max(c));
threshold = c_max * 0.01;
% Apply thresholding
cornerCandidates = c > threshold & c;
% nonmax suppression
result = zeros(size(c));
% check 3x3 neighbourhood

coordinates = [0 0];
for i = 2:height-1
    for j = 2:width-1
        if cornerCandidates(i,j) == 1
            if abs(max(c(i-1: i+1, j-1: j+1), "all")) == abs(c(i,j))
                result(i,j) = 1;
                coordinates = [coordinates; [i j]];
            end
        end
    end
end

%% Return the x and y coordinates of the detected corners


%% Show the image and plot the detected corners
figure;

imshow(img);
hold on;
plot(coordinates(:,1), coordinates(:,2), 'r+', 'MarkerSize', 10, 'LineWidth', 1)
hold off;
