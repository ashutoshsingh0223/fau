function [L,res]=kmeans_dim_entropy_L(X,C,gamma,T,d, eps_C,W,K,eps_Creg);
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
res=sum((X - C*gamma).^2,1);
L=sum(res)*(1/(T*d))+(eps_C/d)*sum(W.*log(max(W,1e-12)));

Ls=0;
for i=1:d
    for k1=1:K
        for k2=1:K
            Ls=Ls+(C(i,k1)-C(i,k2))^2;
        end
    end
end
Ls=1/(T*d*K*(K-1))*Ls;
L=L+eps_Creg*Ls;
end


