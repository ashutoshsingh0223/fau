%% Computation of the PCA dimension reduction for the 
%% data matrix X
%%
%% (C) Illia Horenko, 2018

function [X_proj,V,mu]=MyPCA(X,K);

[V,D]=eigs(cov(X'),K);
mu=mean(X,2);
[N,T]=size(X);
X_proj=V'*(X-repmat(mu,1,T));
X_proj=X_proj;