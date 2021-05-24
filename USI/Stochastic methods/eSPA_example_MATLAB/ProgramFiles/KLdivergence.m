function [ KL ] = KLdivergence(x,y)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
%KL=(x-y)'*(x-y);
%if length(x)~=length(y)
%    length(x)
%    length(y)
%end
[~,ix]=max(x);
[~,iy]=max(y);
KL=-(ix==iy)*size(x,1);
%KL=max(abs(x-y));
%KL=sum(x.*log(max(x./max(y,1e-5),1e-5)));
%KL=0;
%for d=1:length(x)
%    if and(x(d)>0,y(d)>0)
%    KL=KL+0.5*(x(d)*log(x(d))-x(d)*log(y(d)))+...
%        0.5*(y(d)*log(y(d))-y(d)*log(x(d))); 
%    end
%end

