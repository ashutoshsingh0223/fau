function [C]=SPACL_EvaluateCRegularize(X,gamma,K,eps_Creg);
    C=(X*gamma')*inv(gamma*gamma'+2*eps_Creg/(K*(K-1))*(K*eye(K)-ones(K)));
    %C=((gamma*gamma'+2*eps_Creg/(K*(K-1))*(K*eye(K)-ones(K)))\(X*gamma')')';
end
