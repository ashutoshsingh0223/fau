function [C]=SPACL_EvaluateCRegularize_spgqp(X,gamma,K,eps_Creg,W_m,D);
%    C=X*gamma'*inv(gamma*gamma'+2*eps_CL/(K*(K-1))*(K*eye(K)-ones(K)));
    C=((gamma*gamma'+2*eps_Creg/(K*(K-1))*(K*eye(K)-ones(K)))\(X*gamma')')';
    for d=1:D
        if W_m(d)>1e-8
            C(d,:)=(1/W_m(d)).*C(d,:);
        else
            C(d,:)=0*C(d,:);
        end
    end
end
