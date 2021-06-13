function [counter] = gambler(p,n)
%GAMBLER Summary of this function goes here
%   p = Probability of winning
% 1 - p = Probability of losing.
% Number of times he plays the game

% Initializing counter representing number of wins.
%  Since he has to have 1 CHF to play the game we initiliaze the value 1
counter = 1;
    for i=1:n
    % Taking a bernoulli sample with probablity of sucess as p
    r = binornd(1,p);
        if r == 1
%             If gambler wins increment counter by 1 and keep on playing
            counter = counter + 1;
        
        else
%             If gambler loses. He loses everything. So he cant play
%             anymore
            counter = 0;
            return
        end
    end
end

