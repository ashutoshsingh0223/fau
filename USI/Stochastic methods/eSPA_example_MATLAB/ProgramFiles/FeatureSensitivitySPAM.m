function [dGamma_dX,out_opd] = FeatureSensitivitySPAM(out_opd,X,n_eps)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
[N_K,N_ens]=size(out_opd);
[dim,T]=size(X);
i=1;
for k=1:N_K
    for n=1:N_ens
        in{i}=out_opd{k,n};
        in{i}.X=X;
        if nargin==2
        in{i}.C=AllignClusterCenters(out_opd{k,1}.C,out_opd{k,n}.C);
        else
        in{i}.C=AllignClusterCenters(out_opd{k,1}.C_full(:,:,n_eps),out_opd{k,n}.C_full(:,:,n_eps));    
        end
        in{i}.T=T;
        in{i}.d=dim;
        in{i}.K=size(in{i}.C,2);
        in{i}.eps=1e-6;
        kk{i}=k;nn{i}=n;
        i=i+1;
    end
end

parfor n=1:(i-1)
    out{n}=Evaluate_dGamma_dX(in{n});
end

for k=1:N_K
    dGamma_dX{k}=zeros(dim,size(out_opd{k,1}.C,2));
end

for n=1:(i-1)
    dGamma_dX{kk{n}}=dGamma_dX{kk{n}}+(1/N_ens)*out{n}.dGamma_dX;
    out_opd{kk{n},nn{n}}.C=in{n}.C;
    out_opd{kk{n},nn{n}}.gamma_complete=out{n}.gamma;
end
end




