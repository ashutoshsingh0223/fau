function [part1,part2] = bisection_spectral(A,xy,picture)
% bisection_spectral : Spectral partition of a graph.
%
% D.P & O.S for the "HPC Course" at USI and
%                   "HPC Lab for CSE" at ETH Zuric
%
% [part1,part2] = bisection_spectral(A) returns a partition of the n vertices
%                 of A into two lists part1 and part2 according to the
%                 spectral bisection algorithm of Simon et al:
%                 Label the vertices with the components of the Fiedler vector
%                 (the second eigenvector of the Laplacian matrix) and partition
%                 them around the median value or 0.







% Steps
% 1. Construct the Laplacian.
% 2. Calculate its eigensdecomposition.
% 3. Label the vertices with the components of the Fiedler vector.
% 4. Partition them around their median value, or 0.


D = diag(sum(A, 2));
L = D - A;

[V,D] = eigs(L, 2, 1e-12);
v_2 = V(:,2);
med = median(v_2);
part1 = find(v_2 < med);
part2 = find(v_2 >= med);

if picture == 1
%     disp(' ');
%     disp(' HPC Lab at USI:   ');
%     disp(' Implement spectral bisection');
%     disp(' ');
    gplotpart(A,xy,part1);
    title('Spectral bisection using the Fiedler Eigenvector');
end




% <<<< Dummy implementation to generate a partitioning
% n = size(A,1);
% map = zeros(n,1);
% map(1:round((n/2))) = 0; 
% map((round((n/2))+1):n) = 1;
% [part1,part2] = other(map);
% 
% if picture == 1
%     gplotpart(A,xy,part1);
%     title('Spectral bisection (dummy) using the Fiedler Eigenvector');
% end

% Dummy implementation to generate a partitioning >>>>


end
