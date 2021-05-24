function [C]=SPACL_EvaluateCRegularize_analytic(X,gamma,K,d,T);
%    C=(X*gamma')*(1./gamma*gamma'+2*eps_Creg/(K*(K-1))*(K*eye(K)-ones(K)));
C=zeros(d,K);N=sum(gamma',1);
for k=1:K
    for t=1:T
        C(:,k)=C(:,k) + gamma(k,t)*X(:,t);
    end
    if N(k)>0
    C(:,k)=C(:,k)./N(k);
    else
    C(:,k)=(1e-10)*randn(d,1);    
    end
end
%C=((gamma*gamma'+2*eps_Creg/(K*(K-1))*(K*eye(K)-ones(K)))\(X*gamma')')';
end
