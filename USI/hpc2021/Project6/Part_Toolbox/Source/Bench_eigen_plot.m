% Visualize information from the eigenspectrum of the graph Laplacian
%
% D.P & O.S for the "HPC Course" at USI and
%                   "HPC Lab for CSE" at ETH Zurich

% add necessary paths
addpaths_GP;

% Graphical output at bisection level
picture = 0;

% Cases under consideration


% load airfoil1.mat;
% load 3elt.mat;
% load barth4.mat;
% load mesh3e1.mat;
% load crack.mat;

cases = {
    'airfoil1.mat';
    '3elt.mat';
    'barth4.mat';
    'mesh3e1.mat';
    'crack.mat';
    };

% Initialize the cases
for c = 1:length(cases)
    load(cases{c});
    W      = Problem.A;
    coords = Problem.aux.coord;
    
    D = diag(sum(W, 2));
    L = D - W;
    
    [V,D] = eigs(L, 3, 1e-12);
    v_1 = V(:,1);
    v_2 = V(:,2);

% Steps
% 1. Construct the graph Laplacian of the graph in question.
% 2. Compute eigenvectors associated with the smallest eigenvalues.
% 3. Perform spectral bisection.
% 4. Visualize:
%   i.   The first and second eigenvectors.
if strcmp(cases{c}, 'airfoil1.mat')
    x_axis = 1:length(v_1);
    plot(x_axis, v_1,'g', x_axis, v_2, '.');
    legend('EigenVector-1','EigenVector-2');
    pause;
    disp('Hit Space')
end

%   ii.  The second eigenvector projected on the coordinate system space of graphs.
med = median(v_2);
part1 = find(v_2 < med);
part2 = find(v_2 >= med);

coords2 = [coords, v_2] ;
gplotpart(W,coords2,part1);
title(cases{c});
pause;
disp('Hit Space')
%   iii. The spectral bi-partitioning results using the spectral coordinates of each graph.
gplotpart(W,coords,part1);
title(strjoin({cases{c}, 'Spatial Coordinates'}))
disp('Hit Space')
pause;
gplotpart(W,[v_2, V(:,3)],part1);
title(strjoin({cases{c}, 'Spectral Coordinates'}))
disp('Hit Space')
pause;
close all;
end

