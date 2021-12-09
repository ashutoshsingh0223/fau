function [cut_recursive,cut_kway] = Bench_metis(picture)
% Compare recursive bisection and direct k-way partitioning,
% as implemented in the Metis 5.0.2 library.

%  Add necessary paths
addpaths_GP;

% Graphs in question
% load usroads;
usRoads = Initialize_case(load('../Datasets/Roads/usroads.mat'));
% load luxembourg_osm;
lxRoads = Initialize_case(load('../Datasets/Roads/luxembourg_osm.mat'));

fprintf('\n%7s %20s %16s\n','Method','Recursive','Direct');
fprintf('%10s %10d %6d %10d %6d\n','Partitions',16,32,16,32);


% Steps
% 1. Initialize the cases
% 2. Call metismex to
%     a) Recursively partition the graphs in 16 and 32 subsets.
W = usRoads.Adj;
coords = usRoads.coords;
[map,edgecut_us_r_16] = metismex('PartGraphRecursive',W,16);

[map_us_r_32,edgecut_us_r_32] = metismex('PartGraphRecursive',W,32);

[map,edgecut_us_d_16] = metismex('PartGraphKway',W,16);
[map_us_d_32,edgecut_us_d_32] = metismex('PartGraphKway',W,32);



W_lx = lxRoads.Adj;
coords_lx = lxRoads.coords;
[map,edgecut_lx_r_16] = metismex('PartGraphRecursive',W_lx,16);
[map_lx_r_32,edgecut_lx_r_32] = metismex('PartGraphRecursive',W_lx,32);

[map,edgecut_lx_d_16] = metismex('PartGraphKway',W_lx,16);
[map_lx_d_32,edgecut_lx_d_32] = metismex('PartGraphKway',W_lx,32);

if picture == 1
    figure(1)
    gplotmap(W, coords,map_us_r_32)
    title('US Roads 32-Way recursive');
    disp('Hit Space');
    pause;

    figure(2)
    gplotmap(W, coords,map_us_d_32)
    title('US Roads 32-Way direct');
    disp('Hit Space')
    pause;

    figure(3)
    gplotmap(W_lx, coords_lx, map_lx_r_32)
    title('Luxemberg Roads 32-Way recursive');
    disp('Hit Space')
    pause;

    figure(4)
    gplotmap(W_lx, coords_lx, map_lx_d_32)
    title('Luxemberg Roads 32-Way direct');
    disp('Hit Space')
    pause;
    close all;
end



fprintf('%10s %10d %6d %10d %6d\n','US',edgecut_us_r_16,edgecut_us_r_32,edgecut_us_d_16,edgecut_us_d_32);
fprintf('%10s %10d %6d %10d %6d\n','LX',edgecut_lx_r_16,edgecut_lx_r_32,edgecut_lx_d_16,edgecut_lx_d_32);

%     b) Perform direct k-way partitioning of the graphs in 16 and 32 subsets.
% 3. Visualize the results for 32 partitions.


end