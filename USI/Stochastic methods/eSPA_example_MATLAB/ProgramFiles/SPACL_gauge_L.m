function [L]=SPACL_gauge_L(X,pi,C,Lambda,gamma,T,d,m, reg_param, W);
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
L=norm((X - W*C*gamma),'fro')^2*(1/(T*d))-reg_param/(T*m)*sum(sum(pi.*(log(max(Lambda*gamma,1e-12)))));

end


