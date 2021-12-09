% Visualize information from the eigenspectrum of the graph Laplacian
%
% D.P & O.S for the "HPC Course" at USI and
%                   "HPC Lab for CSE" at ETH Zurich

% add necessary paths
addpaths_GP;

% Graphical output at bisection level
picture = 0;

% Cases under consideration
load airfoil1.mat;
% load 3elt.mat;
% load barth4.mat;
% load mesh3e1.mat;
% load crack.mat;

% Initialize the cases
W      = Problem.A;

D = diag(sum(W, 2));
L = D - W;

[V,D] = eigs(L, 2, 1e-12);
v_2 = V(:,2);
med = median(v_2);
part1 = find(v_2 < med);
part2 = find(v_2 >= med);

coords = [Problem.aux.coord, v_2] ;
gplotpart(W,coords,part1);
% Steps
% 1. Construct the graph Laplacian of the graph in question.
% 2. Compute eigenvectors associated with the smallest eigenvalues.
% 3. Perform spectral bisection.
% 4. Visualize:
%   i.   The first and second eigenvectors.
%   ii.  The second eigenvector projected on the coordinate system space of graphs.
%   iii. The spectral bi-partitioning results using the spectral coordinates of each graph.

