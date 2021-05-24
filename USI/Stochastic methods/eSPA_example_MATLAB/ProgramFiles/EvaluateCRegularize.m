function [C,L,L0,norm_C]=EvaluateCRegularize(X,gamma,T,d,K,eps,xt2);
    xt=zeros(d,K);gt=zeros(K);
    for t=1:T
        xt=xt+X(:,t)*gamma(:,t)';
        gt=gt+gamma(:,t)*gamma(:,t)';
    end
    xt=xt./(T*d);
    gt=gt./(T*d);
    %C=xt/(gt+eps/(d*K)*eye(K));
    C=xt/(gt+2*eps/(d*(K-1)*K)*(K*eye(K)-ones(K)));
    L0=0;
    for i=1:d
        for k1=1:K
            for k2=1:K
                L0=L0+(C(i,k1)-C(i,k2))^2;         
            end
%            L0=L0+eps/(d*K)*C(i,k1)^2;
        end
    end
    norm_C=L0/(d*(K-1)*K);
    L0=eps/(d*(K-1)*K)*L0;
    L=-2*trace(xt*C')+trace(C*gt*C')+xt2/(T*d);
    L=L+L0;
end
