% Script to load .csv lists of adjacency matrices and the corresponding 
% coordinates. 
% The resulting graphs should be visualized and saved in a .csv file.
%
% D.P & O.S for the "HPC Course" at USI and
%                   "HPC Lab for CSE" at ETH Zurich

addpaths_GP;

% Steps
% 1. Load the .csv files
data = readtable("Part_Toolbox/Datasets/Countries_Meshes/csv/CH-4468-adj.csv");
% pause;
% 2. Construct the adjaceny matrix (NxN). There are multiple ways
%    to do so.
Adj = sparse(accumarray(data{:,:},1));
Adj = (Adj + transpose(Adj)) / 2;

coords = readtable("Part_Toolbox/Datasets/Countries_Meshes/csv/CH-4468-pts.csv");
coords = coords{:,:};
% pause;
% 3. Visualize the resulting graphs
% gplotg(Adj, coords);
% pause;
% 4. Save the resulting graphs
save('../Datasets/../Datasets/Countries_Mat/Swiss_graph.mat','Adj', 'coords');




% Example of the desired graph format for CH

% load Swiss_graph.mat
% W      = CH_adj;
% coords = CH_coords;   
% whos
% 
% figure;
% gplotg(W,coords);
