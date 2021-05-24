function [Persistent,Mean] = PersistentAndMeanPredictions_Euclidean(X)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[N,T]=size(X);
mean_X=mean(X');
Persistent=0;
Mean=0;
for t=1:T-1
    Persistent=Persistent+sum(norm(X(:,t+1)-X(:,t),2)^2)/(N*(T-1));
    Mean=Mean+sum(norm(X(:,t+1)-mean_X',2)^2)/(N*(T-1));
end
end

