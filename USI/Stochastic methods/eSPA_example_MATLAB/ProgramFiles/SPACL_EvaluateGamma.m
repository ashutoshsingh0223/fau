function [gamma]=SPACL_EvaluateGamma(X,pi,C,Lambda,T,K,m,d,reg_param);
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[~,idx]=min(-(1/m)*reg_param*log(max(Lambda',1e-12))*pi+(1/d)*sqDistance(X, C)');
gamma=zeros(K,T);
for k=1:K
    gamma(k,find(idx==k))=1;
end
end

