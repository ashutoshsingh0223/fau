function [C]=SPACL_dim_entropy_EvaluateCRegularize(X,gamma);
%    C=X*gamma'*inv(gamma*gamma'+2*eps_CL/(K*(K-1))*(K*eye(K)-ones(K)));
    C=((gamma*gamma')\(X*gamma')')';
end
