function [out_opd] = SPACL_l1(X,pi,K,N_anneal,out_init,reg_param,X_valid,pi_valid,N_neurons,flag_AUC)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[d,T]=size(X);m=size(pi,1);
xt2=0;pit2=0;xt2_valid=0;
for t=1:T
    xt2=xt2+X(:,t)'*X(:,t);
    pit2=pit2+pi(:,t)'*pi(:,t);
end
for t=1:size(X_valid,2)
    xt2_valid=xt2_valid+X_valid(:,t)'*X_valid(:,t);
end
%xt2=xt2/(T*d);

kkk=1;
for n=1:N_anneal
    gamma=rand(K,T);
    for t=1:T
        gamma(:,t)=gamma(:,t)./sum(gamma(:,t));
    end
    if n==1
        gamma=out_init.gamma;
    end
    Lambda=rand(m,K);
    for k=1:K
         Lambda(:,k)=Lambda(:,k)./sum(Lambda(:,k));
    end
    C=rand(d,K);
    if n==1
        C=out_init.C_fin;
    end
         %     if n==2
%         T_switch=1:ceil(T/K):T;
%         if T_switch(length(T_switch))~=T
%             T_switch=[T_switch T];
%         end
%         gamma=zeros(K,T);
%         for t=1:min(length(T_switch)-1,K)
%             gamma(t,T_switch(t):T_switch(t+1)-1)=1;
%         end
%         gamma(t,T)=1;
%     end
    for e=1:size(reg_param,2)
        in{kkk}.X=X;
        in{kkk}.flag_AUC=flag_AUC;
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
        in{kkk}.Lambda=Lambda;
        in{kkk}.C=C;
        in{kkk}.reg_param=reg_param(:,e);
        in{kkk}.e=e;
        in{kkk}.n=n;
        in{kkk}.N_anneal=N_anneal;
        in{kkk}.N_neurons=N_neurons;
        kkk=kkk+1;
    end
end
parfor kkk=1:numel(in)
    [out{kkk}]=SPACL_Replica(in{kkk});
end
for kkk=1:numel(in)
    n=in{kkk}.n;
    e=in{kkk}.e;
    L_discr_full(n,e)=out{kkk}.L_discr_train;
    %L(n,e)=out{kkk}.L_fin;
    L_pred(n,e)=out{kkk}.L_pred_Markov;
    KKK(n,e)=kkk;
end
for e=1:size(L_pred,2)
    %[lll(e),ii]=min(L_discr_full(:,e));
    [lll(e),ii]=min(L_pred(:,e));
    %L_fin(e)=out{KKK(ii,e)}.L;
    L_fin(e)=out{KKK(ii,e)}.L;
    L_pred_valid(e)=out{KKK(ii,e)}.L_pred_valid;
    L_pred_test(e)=out{KKK(ii,e)}.L_pred_test;
    L_discr_valid(e)=out{KKK(ii,e)}.L_discr_valid;
    L_discr_train(e)=out{KKK(ii,e)}.L_discr_train;
    L_pred_Markov(e)=out{KKK(ii,e)}.L_pred_Markov;
    gamma_fin(:,:,e)=out{KKK(ii,e)}.gamma;
    gamma_valid_fin(:,:,e)=out{KKK(ii,e)}.gamma_valid;
    P(:,:,e)=out{KKK(ii,e)}.P;
    net{e}=out{KKK(ii,e)}.net;
    C_fin(:,:,e)=out{KKK(ii,e)}.C;
    norm_C(e,:)=out{KKK(ii,e)}.norm_C;
end
[~,e]=min(lll);
out_opd.gamma=gamma_fin(:,:,e);
out_opd.gamma_valid=gamma_valid_fin(:,:,e);
out_opd.C_full=C_fin;
out_opd.L_full=L_fin;
out_opd.C=C_fin(:,:,e);
out_opd.net=net{e};
out_opd.L=L_fin(e);
out_opd.L_discr_valid=L_discr_valid(e);
out_opd.L_pred_valid=L_pred_valid(e);
out_opd.L_pred_Markov=L_pred_Markov(e);
out_opd.P=P(:,:,e);
out_opd.L_discr_train=L_discr_train;
out_opd.norm_C=norm_C;
out_opd.reg_param=reg_param(e);
end

function [out]=SPACL_Replica(in)
    eps_C=in.reg_param;
    X=in.X;
    N_neurons=in.N_neurons;
    gamma=in.gamma;
    C=in.C;
    X_valid=in.X_valid;
    pi_valid=in.pi_valid;
    T=in.T;
    K=in.K;
    N_anneal=in.N_anneal;
    pi=in.pi;
    flag_AUC=in.flag_AUC;
    %GO=in.GO;
    Lambda=in.Lambda;
    m=size(pi,1);
    d=in.d;
    xt2=in.xt2;
    xt2_valid=in.xt2_valid;
    pit2=in.pit2;
    reg_param=0.01;
    i=1;
    delta_L=1e10;eps=1e-8;
    MaxIter=2000;
    L=[];
%     H2=zeros(d*K,d*K,size(GO,3));
%     for i=1:size(GO,3)
%         for i1=1:K
%             H2((i1-1)*d+1:i1*d,(i1-1)*d+1:i1*d,i)=1/(d^2*K)*(diag(sum(GO(:,:,i),1))+diag(sum(GO(:,:,i),2))-2*GO(:,:,i));
%         end
%     end
    H=zeros(K*(K-1)*d,K*d);
    kk=1;
    for k1=1:K
        for k2=setdiff(1:K,k1)
            for i1=1:d
                H(kk,(k1-1)*d+i1)=1;
                H(kk,(k2-1)*d+i1)=-1;
                kk=kk+1;
            end
        end
    end
    while and(delta_L>eps,i<=MaxIter)
       %[C]=SPACL_EvaluateCRegularize_l1(C,X,gamma,eps_C,K,d,H);
         [C]=SPACL_EvaluateCRegularize(X,gamma,eps_C,K);
      % [L_3]=SPACL_L(X,pi,C,Lambda,gamma,T,d,m,K,  reg_param, eps_C)
        [gamma]=SPACL_EvaluateGamma(X,pi,C,Lambda,T,K,m,d,reg_param);
       %[L_3]=SPACL_L(X,pi,C,Lambda,gamma,T,d,m,K,  reg_param, eps_C)
      [Lambda]=SPACL_EvaluateLambdaRegularize(pi,gamma,m,K);
         %[gamma,L3]=EvaluateGammaParallel_Classify(X,pi,C,P,T,d,K,gamma,xt2,pit2,reg_param);
       %[L_3,Lfin,norm_C]=SPACL_plus_L_l1(X,pi,C,Lambda,gamma,T,d,m,K,  reg_param, eps_C,H);
       [L_3,Lfin,norm_C]=SPACL_L(X,pi,C,Lambda,gamma,T,d,m,K,  reg_param, eps_C);
       %L_3
         %[gamma,L3]=EvaluateGammaParallel_Classify(X,pi,C,P,T,d,K,gamma,xt2,pit2,reg_param);
        L=[L L_3];
        if i>1 
        delta_L=(L(i-1)-L(i));
        end
        %x=reshape(C,K*d,1);sum(abs(H*x))
        %figure(11); mesh(C);pause(0.05)
        %figure(12);plot(L);pause(0.05)
     i=i+1;
    end
    T_valid=size(X_valid,2);
    gamma1{1}=gamma;pi1{1}=pi;
    P=Lambda;%lambdasolver_quadprog_Classify(gamma1,pi1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%    [gamma_valid,LLL2]=EvaluateGammaParallel(X_valid,C,T_valid,d,K,[],xt2_valid);
    [gamma_valid]=SPACL_EvaluateGamma_valid(X_valid,C,T_valid,K);
%    [gamma_valid,LLL2]=EvaluateGammaParallel_Lukas_vect(X_valid,C,T_valid,d,K,[],xt2_valid);
   
   [L_pred_Markov] = AUC_of_Prediction(gamma_valid,P,pi_valid,0,flag_AUC);

    L_pred_Markov=L_pred_Markov/(T_valid*m);
    [L_pred_Markov_train] = AUC_of_Prediction(gamma,P,pi,0,flag_AUC);
    L_pred_Markov_train=L_pred_Markov_train/(T*m);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for n=1:N_anneal
        net = patternnet(N_neurons);
        net.trainParam.showWindow = 0;
        net = train(net,gamma,pi,'useParallel','no');
        err=0;
        for t=1:T
            err=err+KLdivergence(pi(:,t),net(gamma(:,t)));
        end
        err=err/(m*T);
        if n==1
            LLL=err;
            net_final=net;
        else
            if LLL>err
                LLL=err;
                net_final=net;
            end
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %net = patternnet(N_neurons);net.trainParam.showWindow = 0;net = train(net,gamma,pi,'useParallel','no');    
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    net=net_final;
      L_discr_train=0;
    for t=1:T
        dist=X(:,t)-C*gamma(:,t);
        L_discr_train=L_discr_train+dist'*dist;
    end
    L_discr_train=L_discr_train/(T*d);
   [L_pred_test] = AUC_of_Prediction(gamma,net,pi,1,flag_AUC);
    L_pred_test=L_pred_test/(T*m);
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   [L_pred_valid] = AUC_of_Prediction(gamma_valid,net,pi_valid,1,flag_AUC);
    L_pred_valid=L_pred_valid/(T_valid*m);
    L_discr_valid=0;
    for t=1:T_valid
        dist=X_valid(:,t)-C*gamma_valid(:,t);
        L_discr_valid=L_discr_valid+dist'*dist;
    end
    L_discr_valid=L_discr_valid/(T_valid*d);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    out.L=Lfin;
    out.L_discr_train=L_discr_train;
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
    out.reg_param=reg_param;
end

