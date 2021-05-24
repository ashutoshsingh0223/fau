function [] = VisualizeClusters2d(X,out,ind_K,ind_reg,dim)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
[K,T]=size(out{ind_K}.gamma{ind_reg});
figure;hold on
col='rgbcmykrgbcmykrgbcmykrgbcmykrgbcmykrgbcmyk';
mark='xo<>.xo<>.xo<>.xo<>.xo<>.xo<>.xo<>.xo<>.xo<>.xo<>.xo<>.xo<>.';
for k=1:K
    ii=find(out{ind_K}.gamma{ind_reg}(k,:)==1);
    plot(X(dim(1),ii),X(dim(2),ii),[col(k) mark(k)]);
end
end

