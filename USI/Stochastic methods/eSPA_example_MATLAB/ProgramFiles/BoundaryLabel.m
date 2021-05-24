function [Boundary,data] = BoundaryLabel(label,data)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
[NX,NY,NZ]=size(label);
mask=ones(3,3,3);
Boundary=0;
for i=1:6
bin=convn(double(label==i),mask,'valid')/27;%bin=bin(2:NX+1,2:NY+1,2:NZ+1);
flag=double((bin<1)&(bin>(14/27)));
Boundary=Boundary+i*flag;
Label_filt=flag*i;
end
data=data(2:NX-1,2:NY-1,2:NZ-1);
end

