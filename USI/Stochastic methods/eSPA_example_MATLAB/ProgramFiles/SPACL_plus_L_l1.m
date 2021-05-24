function [Lfin,L,Ls]=SPACL_plus_L_l1(X,pi,C,Lambda,gamma,T,d,m,K, reg_param, eps_C,H);
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
L=norm(X - C*gamma,'fro')^2*1/(T*d)-reg_param/(T*m)*sum(sum(pi.*(log(max(Lambda*gamma,1e-12)))));
Ls=sum(abs(H*reshape(C,d*K,1)));
% Ls=0;
% for i=1:d
%     for k1=1:K
%         for k2=1:K
%             Ls=Ls+absC(i,k1)-C(i,k2));
%         end
%     end
% end
Ls=1/(T*d*K*(K-1))*Ls;
Lfin=L+eps_C*Ls;
end


