searchrange = 4;
blocksize = 8;

s = imread('Frame1.png');
g = imread('Frame2.png');
m=1;
n=1;
motion = zeros(size(s,1)/blocksize,size(s,2)/blocksize,2);


for block_y = 1:size(s,1)/blocksize
    for block_x = 1:size(s,2)/blocksize
        m = 1+(block_y-1)*blocksize;
        n = 1+(block_x-1)*blocksize;
        ssd = zeros(2*searchrange+1);
        ssd(:,:) = Inf;
        for k = -searchrange: searchrange
            for l= -searchrange: searchrange
%                 First two if statements for preventing out-of-index on
%                 lower side (m+k) won't exist in the block if |k| > m and k is
%                 negative
%                 the last two if statements to prevent same thing on other
%                 side i.e m+k index won't exist in the current block
%                 (m+blocksize+k)<=size(g,1) since m is the centre of block
%                 not image.
                if (m+k)>0 && (n+l)>0 && (m+blocksize+k)<=size(g,1) && (n+blocksize+l)<=size(g,2)
                    ssd(k+searchrange+1,l+searchrange+1) = calc_ssd(s(m : m + blocksize - 1, n: n + blocksize - 1), g(m+k: m + blocksize + k - 1,  n+l: n + blocksize + l - 1 ));         
                end
            end
        end
        [~, min_rows] = min(ssd);
        [~, min_col] = min(min(ssd));
        motion(block_y, block_x, 1) = min_rows(min_col) - searchrange - 1;
        motion(block_y, block_x, 2) = min_col - searchrange - 1;

        % Determine the motion vector for each block at the indicated position
        % To Do 2.1.4
    end
end

block_num_horiz = size(motion, 2);
block_num_vert = size(motion, 1);

[X, Y] = meshgrid(blocksize/2:blocksize:block_num_horiz*blocksize, blocksize/2:blocksize:block_num_vert*blocksize);
imshow(s)
hold on;

quiver(X, Y, motion(:,:,2), motion(:,:,1))

% hold off;
%% Show the first frame of the sequence and overlay it with a plot of the resulting motion vectors.
% To Do 2.1.5

