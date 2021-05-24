function [out_opd] = ConvMeansParallelRegularize_Classify(X,pi,K,N_anneal,gamma_init,reg_param,X_valid,pi_valid,N_neurons)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[d,T]=size(X);
xt2=0;pit2=0;xt2_valid=0;
for t=1:T
    xt2=xt2+X(:,t)'*X(:,t);
    pit2=pit2+pi(:,t)'*pi(:,t);
end
for t=1:size(X_valid,2)
    xt2_valid=xt2_valid+X_valid(:,t)'*X_valid(:,t);
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
        in{kkk}.pi=pi;
        in{kkk}.X_valid=X_valid;
        in{kkk}.pi_valid=pi_valid;
        in{kkk}.T=T;
        in{kkk}.K=K;
        in{kkk}.d=d;
        in{kkk}.xt2=xt2;
        in{kkk}.xt2_valid=xt2_valid;
        in{kkk}.pit2=pit2;
        in{kkk}.reg_param=reg_param(e);
        in{kkk}.e=e;
        in{kkk}.n=n;
        in{kkk}.N_neurons=N_neurons;
        kkk=kkk+1;
    end
end
parfor kkk=1:numel(in)
    [out{kkk}]=KHullsAlgorithmReplica_Classify(in{kkk});
end
for kkk=1:numel(in)
    n=in{kkk}.n;
    e=in{kkk}.e;
    L(n,e)=out{kkk}.L_pred_test;
    %L_pred(n,e)=out{kkk}.L_pred;
    KKK(n,e)=kkk;
end
for e=1:size(L,2)
    [~,ii]=min(L(:,e));
    %Lfin(e)=out{KKK(ii,e)}.L;
    L_fin(e)=out{KKK(ii,e)}.L;
    L_pred_valid(e)=out{KKK(ii,e)}.L_pred_valid;
    L_pred_test(e)=out{KKK(ii,e)}.L_pred_test;
    L_discr_valid(e)=out{KKK(ii,e)}.L_discr_valid;
    L_pred_Markov(e)=out{KKK(ii,e)}.L_pred_Markov;
    gamma_fin(:,:,e)=out{KKK(ii,e)}.gamma;
    gamma_valid_fin(:,:,e)=out{KKK(ii,e)}.gamma_valid;
    P(:,:,e)=out{KKK(ii,e)}.P;
    net{e}=out{KKK(ii,e)}.net;
    C_fin(:,:,e)=out{KKK(ii,e)}.C;
    norm_C(e)=out{KKK(ii,e)}.norm_C;
end
[~,e]=min(L_pred_test);
out_opd.gamma=gamma_fin(:,:,e);
out_opd.gamma_valid=gamma_valid_fin(:,:,e);
out_opd.C=C_fin(:,:,e);
out_opd.net=net{e};
out_opd.L=L_fin(e);
out_opd.L_discr_valid=L_discr_valid(e);
out_opd.L_pred_valid=L_pred_valid(e);
out_opd.L_pred_Markov=L_pred_Markov(e);
out_opd.P=P(:,:,e);
out_opd.norm_C=norm_C(e);
end

function [out]=KHullsAlgorithmReplica_Classify(in)
    X=in.X;
    N_neurons=in.N_neurons;
    gamma=in.gamma;
    X_valid=in.X_valid;
    pi_valid=in.pi_valid;
    T=in.T;
    K=in.K;
    pi=in.pi;
    m=size(pi,1);
    d=in.d;
    xt2=in.xt2;
    xt2_valid=in.xt2_valid;
    pit2=in.pit2;
    reg_param=in.reg_param;
    i=1;
    delta_L=1e10;eps=1e-5;
    MaxIter=100;
    L=[];
    while and(delta_L>eps,i<=MaxIter)
        [C,L1,L0,norm_C]=EvaluateCRegularize(X,gamma,T,d,K,reg_param,xt2);
        %gamma1{1}=gamma;pi1{1}=pi; 
        %P=lambdasolver_quadprog_Classify(gamma1,pi1);
        [gamma,L2]=EvaluateGammaParallel_Lukas(X,C,T,d,K,gamma,xt2);
        %[gamma,L2]=EvaluateGammaParallel(X,C,T,d,K,gamma,xt2);

        %[gamma,L3]=EvaluateGammaParallel_Classify(X,pi,C,P,T,d,K,gamma,xt2,pit2,reg_param);
        L=[L L2+L0];
        if i>1
        delta_L=(L(i-1)-L(i));
        end
        i=i+1;
    end
    T_valid=size(X_valid,2);
    gamma1{1}=gamma;pi1{1}=pi;
    P=lambdasolver_quadprog_Classify(gamma1,pi1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [gamma_valid,L2]=EvaluateGammaParallel(X_valid,C,T_valid,d,K,[],xt2_valid);
    L_pred_Markov=0;
    for t=1:T_valid
        L_pred_Markov=L_pred_Markov+KLdivergence(pi_valid(:,t),P*gamma_valid(:,t));
    end
    L_pred_Markov=L_pred_Markov/(T_valid*m);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    net = patternnet(N_neurons);net.trainParam.showWindow = 0;net = train(net,gamma,pi,'useParallel','no');    
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    L_pred_test=0;
    for t=1:T
        L_pred_test=L_pred_test+KLdivergence(pi(:,t),net(gamma(:,t)));
    end
    L_pred_test=L_pred_test/(T*m);
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    L_pred_valid=0;
    for t=1:T_valid
        L_pred_valid=L_pred_valid+KLdivergence(pi_valid(:,t),net(gamma_valid(:,t)));
    end
    L_pred_valid=L_pred_valid/(T_valid*m);
    L_discr_valid=0;
    for t=1:T_valid
        dist=X_valid(:,t)-C*gamma_valid(:,t);
        L_discr_valid=L_discr_valid+dist'*dist;
    end
    L_discr_valid=L_discr_valid/(T_valid*d);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    out.L=L2;
    out.L_discr_valid=L_discr_valid;
    out.L_pred_valid=L_pred_valid;
    out.L_pred_Markov=L_pred_Markov;
    out.P=P;
    out.L_pred_test=L_pred_test;
    out.norm_C=norm_C;
    out.gamma=gamma;
    out.gamma_valid=gamma_valid;
    out.C=C;
    out.net=net;
end

