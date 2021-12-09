% Benchmark for recursively partitioning meshes, based on various
% bisection approaches
%
% D.P & O.S for the "HPC Course" at USI and
%                   "HPC Lab for CSE" at ETH Zurich



% add necessary paths
addpaths_GP;
nlevels_a = 3;
nlevels_b = 4;

fprintf('       *********************************************\n')
fprintf('       ***  Recursive graph bisection benchmark  ***\n');
fprintf('       *********************************************\n')

% load cases
cases = {
    'airfoil1.mat';
    'netz4504_dual.mat';
    'stufe.mat';
    '3elt.mat';
    'barth4.mat';
    'ukerbe1.mat';
    'crack.mat';
    };

nc = length(cases);
maxlen = 0;
for c = 1:nc
    if length(cases{c}) > maxlen
        maxlen = length(cases{c});
    end
end

for c = 1:nc
    fprintf('.');
    sparse_matrices(c) = load(cases{c});
end


fprintf('\n\n Report Cases         Nodes     Edges\n');
fprintf(repmat('-', 1, 40));
fprintf('\n');
for c = 1:nc
    spacers  = repmat('.', 1, maxlen+3-length(cases{c}));
    [params] = Initialize_case(sparse_matrices(c));
    fprintf('%s %s %10d %10d\n', cases{c}, spacers,params.numberOfVertices,params.numberOfEdges);
end

%% Create results table
fprintf('\n%7s %16s %20s %16s %16s\n','Bisection','Spectral','Metis 5.0.2','Coordinate','Inertial');
fprintf('%10s %10d %6d %10d %6d %10d %6d %10d %6d\n','Partitions',8,16,8,16,8,16,8,16);
fprintf(repmat('-', 1, 100));
fprintf('\n');


for c = 1:nc
    spacers = repmat('.', 1, maxlen+3-length(cases{c}));
    fprintf('%s %s', cases{c}, spacers);
    sparse_matrix = load(cases{c});
    

    % Recursively bisect the loaded graphs in 8 and 16 subgraphs.
    % Steps
    % 1. Initialize the problem
    [params] = Initialize_case(sparse_matrices(c));
    W      = params.Adj;
    coords = params.coords;
    picture = 0;
    if strcmp(cases{c},'crack.mat') == 1
        picture = 0;
    end
    [map,sepij,sepA] = rec_bisection('bisection_spectral', 3, W, coords, 0);
    [cut_spectral_8] = cutsize(W,map);
     if picture == 1
       figure(1)
       gplotmap(W, coords, map)
       title('Bisection Spectral-8 crack.mat');
       disp('Hit Space to continue');
       pause;
     end

    [map,sepij,sepA] = rec_bisection('bisection_spectral', 4, W, coords, 0);
    [cut_spectral_16] = cutsize(W,map);
     if picture == 1
       figure(1)
       gplotmap(W, coords, map)
       title('Bisection Spectral-16 crack.mat');
       disp('Hit Space to continue');
       pause;
     end

    [map,sepij,sepA] = rec_bisection('bisection_metis', 3, W, coords, 0);
    [cut_metis_8] = cutsize(W,map);
     if picture == 1
       figure(3)
       gplotmap(W, coords, map)
       title('Bisection Metis-8 crack.mat');
       disp('Hit Space to continue');
       pause;
     end

    [map,sepij,sepA] = rec_bisection('bisection_metis', 4, W, coords, 0);
    [cut_metis_16] = cutsize(W,map);
     if picture == 1
       figure(4)
       gplotmap(W, coords, map)
       title('Bisection Metis-16 crack.mat');
       disp('Hit Space to continue');
       pause;
     end

    [map,sepij,sepA] = rec_bisection('bisection_coordinate', 3, W, coords, 0);
    [cut_coordinate_8] = cutsize(W,map);
     if picture == 1
       figure(5)
       gplotmap(W, coords, map)
       title('Bisection Coordinate-8 crack.mat');
       disp('Hit Space to continue');
       pause;
     end

    [map,sepij,sepA] = rec_bisection('bisection_coordinate', 4, W, coords, 0);
    [cut_coordinate_16] = cutsize(W,map);
     if picture == 1
       figure(6)
       gplotmap(W, coords, map)
       title('Bisection Coordinate-16 crack.mat');
       disp('Hit Space to continue');
       pause;
     end

    [map,sepij,sepA] = rec_bisection('bisection_inertial', 3, W, coords, 0);
    [cut_interial_8] = cutsize(W,map);
     if picture == 1
       figure(7)
       gplotmap(W, coords, map)
       title('Bisection Intertial-8 crack.mat');
       disp('Hit Space to continue');
       pause;
     end

    [map,sepij,sepA] = rec_bisection('bisection_inertial', 4, W, coords, 0);
    [cut_interial_16] = cutsize(W,map);
     if picture == 1
       figure(8)
       gplotmap(W, coords, map)
       title('Bisection Intertial-16 crack.mat');
       disp('Hit Space to continue');
       pause;
       close all;
     end


    % 2. Recursive routines
    % i. Spectral    
    % ii. Metis
    % iii. Coordinate    
    % iv. Inertial
    % 3. Calculate number of cut edges
    % 4. Visualize the partitioning result
    
    
    fprintf('%6d %6d %10d %6d %10d %6d %10d %6d\n',cut_spectral_8,cut_spectral_16, ...
        cut_metis_8, cut_metis_16, cut_coordinate_8, cut_coordinate_16, cut_interial_8, cut_interial_16);
    
end
