%
% D.P & O.S for the "HPC Course" at USI and
%                   "HPC Lab for CSE" at ETH Zurich

function [part1,part2] = bisection_inertial(A,xy,picture)
% bisection_inertial : Inertial partition of a graph.
%
% [p1,p2] = bisection_inertial(A,xy) returns a list of the vertices on one side of a partition
%     obtained by bisection with a line or plane normal to a moment of inertia
%     of the vertices, considered as points in Euclidean space.
%     Input A is the adjacency matrix of the mesh (used only for the picture!);
%     each row of xy is the coordinates of a point in d-space.
%
% bisection_inertial(A,xy,1) also draws a picture.


% disp(' ');
% disp(' HPC Lab at USI:   ');
% disp(' Implement inertial bisection');
% disp(' ');


% Steps
% 1. Calculate the center of mass.
% 2. Construct the matrix M.
%  (Consult the pdf of the assignment for the creation of M) 
% 3. Calculate the smallest eigenvector of M.  
% 4. Find the line L on which the center of mass lies.
% 5. Partition the points around the line L.
%   (you may use the function partition.m)
pt = mean(xy, 1);
xx = xy(:,1) - pt(1);
yy = xy(:,2) - pt(2);

Sxx = sum(pow2(xx));
Syy = sum(pow2(yy));
Sxy = sum(xx .* yy);

M = [Syy, Sxy;Sxy, Sxx];
[V,D] = eig(M);
u = V(:,1);

[part1, part2] = partition(xy, u);

% <<<< Dummy implementation to generate a partitioning
% n   = size(A,1);
% map = zeros(n,1);
% map(1:round((n/2)))     = 0; 
% map((round((n/2))+1):n) = 1;
% 
% 
% [part1,part2] = other(map);

if picture == 1
    gplotpart(A,xy,part1);
    title('Inertial bisection using the Fiedler Eigenvector');
end

% Dummy implementation to generate a partitioning >>>>

end
