function [L]=SPACL_dim_entropy_L_v2(X,pi,C,Lambda,gamma,T,d,m, reg_param, eps_C,W,K,eps_Creg);
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
L=norm(diag(sqrt(W))*(X - C*gamma),'fro')^2*(1/(T*d))+(eps_C/d)*sum(W.*log(max(W,1e-12)))-reg_param/(T*m)*sum(sum(pi.*(log(max(Lambda*gamma,1e-12)))));

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


