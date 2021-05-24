function [Persistent,Mean] = PersistentAndMeanPredictions(X)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[N,T]=size(X);
mean_X=mean(X');
Persistent=0;
Mean=0;
for t=1:T-1
    Persistent=Persistent+sum(abs(X(:,t+1)-X(:,t)))/(N*(T-1));
    Mean=Mean+sum(abs(X(:,t+1)-mean_X'))/(N*(T-1));
end
end

