function [gamma]=SPACL_EvaluateGamma_valid(X,C,T,K)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[~,idx]=min(sqDistance(X, C)');
gamma=zeros(K,T);
for k=1:K
    gamma(k,find(idx==k))=1;
end
end

