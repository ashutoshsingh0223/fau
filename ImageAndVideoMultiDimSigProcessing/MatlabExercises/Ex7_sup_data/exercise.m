
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
for t=1:50
%   fprintf('./RaceHorses_832x480/RaceHorses_832x480_30_orig_f%d.png \n',n);

images{t} = imread(sprintf('./RaceHorses_416x240/RaceHorses_416x240_30_orig_f%d.png',t));
end

%   Calculate time derivative  
for t=49:50
    s_t = images{t} - images{t-1};

    for x=2:size(images{t-1}, 1)
        s_x(x, :) = images{t-1}(x, :) - images{t-1}(x-1, :);
    end

    for y=2:size(images{t-1}, 2)
        s_y(:, y) = images{t-1}(:, y) - images{t-1}(:, y-1);
    end
    
    
    motion = zeros(size(images{t-1}, 1), size(images{t-1}, 2), 2);

    for i=2:size(images{t-1}, 1) - 1
        for j=2:size(images{t-1}, 2) - 1
            S = zeros(9, 2);
            sx_3_3 = s_x(i-1:i+1, j-1:j+1);
            sy_3_3 = s_y(i-1:i+1, j-1:j+1);
            
            st_3_3 = s_t(i-1:i+1, j-1:j+1);

            S(:,1) = sx_3_3(:);
            S(:,2) = sy_3_3(:);
            STS = transpose(S)* S;
            if rank(STS, 1e-8) == size(STS,1)
                motion(i,j, :) = inv(STS) * transpose(double(S)) * double(st_3_3(:));
            end
        end
    end
end


for i=1:50
    race_horses_small(:, :, i) = imread(['./RaceHorses_416x240/RaceHorses_416x240_30_orig_f' num2str(i) '.png']);
    race_horses_large(:, :, i) = imread(['./RaceHorses_832x480/RaceHorses_832x480_30_orig_f' num2str(i) '.png']);

end

opticalFlow = opticalFlowLK();
flow = estimateFlow(opticalFlow, race_horses_small(:,:,1));
imshow(race_horses_small(:,:,1));

for i=2:50
    imshow(race_horses_small(:,:,i - 1));
    hold on;
    plot(flow, 'DecimationFactor', [5 5], 'ScaleFactor', 10);
    hold off;
    flow = estimateFlow(opticalFlow, race_horses_small(:,:,i));
    pause(0.5);
end

reset(opticalFlow);
% Start again with horses large


