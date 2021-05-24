function [gamma_fin,C_fin,Lfin,norm_C] = ConvMeansParallelRegularize(X,K,N_anneal,gamma_init,reg_param)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[d,T]=size(X);
xt2=0;
for t=1:T
    xt2=xt2+X(:,t)'*X(:,t);
end
kkk=1;
for n=1:N_anneal
    gamma=rand(K,T);
    for t=1:T
        gamma(:,t)=gamma(:,t)./sum(gamma(:,t));
    end
    if n==1
        gamma=gamma_init;
    end
    if n==2
        T_switch=1:ceil(T/K):T;
        if T_switch(length(T_switch))~=T
            T_switch=[T_switch T];
        end
        gamma=zeros(K,T);
        for t=1:min(length(T_switch)-1,K)
            gamma(t,T_switch(t):T_switch(t+1)-1)=1;
        end
        gamma(t,T)=1;
    end
    for e=1:length(reg_param)
        in{kkk}.X=X;
        in{kkk}.gamma=gamma;
        in{kkk}.T=T;
        in{kkk}.K=K;
        in{kkk}.d=d;
        in{kkk}.xt2=xt2;
        in{kkk}.reg_param=reg_param(e);
        in{kkk}.e=e;
        in{kkk}.n=n;
        kkk=kkk+1;
    end
end
parfor kkk=1:numel(in)
    [out{kkk}]=KHullsAlgorithmReplica(in{kkk});
end
for kkk=1:numel(in)
    n=in{kkk}.n;
    e=in{kkk}.e;
    L(n,e)=out{kkk}.L;
    KKK(n,e)=kkk;
end
for e=1:size(L,2)
    [Lfin(e),ii]=min(L(:,e));
    gamma_fin(:,:,e)=out{KKK(ii,e)}.gamma;
    C_fin(:,:,e)=out{KKK(ii,e)}.C;
    norm_C(e)=out{KKK(ii,e)}.norm_C;
end
end

function [out]=KHullsAlgorithmReplica(in)
    X=in.X;
    gamma=in.gamma;
    T=in.T;
    K=in.K;
    d=in.d;
    xt2=in.xt2;
    reg_param=in.reg_param;
    i=1;
    delta_L=1e10;eps=1e-5;
    MaxIter=100;
    L=[];
    while and(delta_L>eps,i<=MaxIter)
        [C,L1,L0,norm_C]=EvaluateCRegularize(X,gamma,T,d,K,reg_param,xt2);
        [gamma,L2]=EvaluateGammaParallel(X,C,T,d,K,gamma,xt2);
        L=[L L2+L0];
        if i>1
        delta_L=(L(i-1)-L(i));
        end
        i=i+1;
    end
    out.L=L2;
    out.norm_C=norm_C;
    out.gamma=gamma;
    out.C=C;
end

