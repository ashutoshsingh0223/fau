function [out_opd] = SPACL_kmeans_dim_entropy_spgqp(X,pi,K,N_anneal,out_init,reg_param,X_valid,pi_valid,N_neurons,flag_AUC)
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
    if and(~isempty(out_init),n==1)
        gamma=out_init.gamma;
    end
    Lambda=rand(m,K);
    for k=1:K
         Lambda(:,k)=Lambda(:,k)./sum(Lambda(:,k));
    end
    if and(~isempty(out_init),n==1)
        Lambda=out_init.P;
    end
    C=2*rand(d,K)-1;
    W=rand(1,d);W=W./sum(W);%ones(1,d)./d;
%    if n==1
%        C=out_init.C;
%    end
    
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
        in{kkk}.W=W;
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
for kkk=1:numel(in)
    [out{kkk}]=SPACL_Replica(in{kkk});
end
for kkk=1:numel(in)
    n=in{kkk}.n;
    e=in{kkk}.e;
    L_discr_full(n,e)=out{kkk}.L_discr_train;
    %L(n,e)=out{kkk}.L_fin;
    L_pred(n,e)=out{kkk}.L_pred_Markov;
    KKK(n,e)=kkk;
    time(n,e)=out{kkk}.time;
    N_params(n,e)=out{kkk}.N_params;
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
    W_fin(:,e)=out{KKK(ii,e)}.W';
    C_fin(:,:,e)=out{KKK(ii,e)}.C;
    %norm_C(e,:)=out{KKK(ii,e)}.norm_C;
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
%out_opd.L_pred_valid=L_pred_valid(e);
out_opd.L_pred_valid_Markov=L_pred_Markov(e);
out_opd.P=P(:,:,e);
out_opd.L_discr_train=L_discr_train;
%out_opd.norm_C=norm_C;
out_opd.reg_param_W=reg_param(1,e);
out_opd.reg_param_CL=reg_param(2,e);
out_opd.W=W_fin(:,e);
out_opd.time_Markov=mean(mean(time));
out_opd.N_params_Markov=mean(mean(N_params));
end

function [out]=SPACL_Replica(in)
    eps_C=in.reg_param(1);
    X=in.X;
    N_neurons=in.N_neurons;
    gamma=in.gamma;
    C=in.C;
    W=in.W;
    X_valid=in.X_valid;
    pi_valid=in.pi_valid;
    T=in.T;
    K=in.K;
    N_anneal=in.N_anneal;
    pi=in.pi;
    flag_AUC=in.flag_AUC;
    Lambda=in.Lambda;
    m=size(pi,1);
    d=in.d;
    xt2=in.xt2;
    xt2_valid=in.xt2_valid;
    pit2=in.pit2;
    reg_param=in.reg_param(2);
    i=1;
    delta_L=1e10;eps=1e-8;
    eps_Creg=1e-10;
    MaxIter=100;
    L=[];
    tic;
   
    while and(delta_L>eps,i<=MaxIter)
      %[L_3]=SPACL_dim_entropy_L(X,pi,C,Lambda,gamma,T,d,m,  reg_param, eps_C,W);
      % L_3
      W_m=sqrt(W)';
      X_W=bsxfun(@times,X,W_m);
      [gamma]=SPACL_EvaluateGamma(X_W,pi,bsxfun(@times,C,W_m),Lambda,T,K,m,d,reg_param);
      %[gamma]=SPACL_EvaluateGamma(diag(sqrt(W))*X,pi,diag(sqrt(W))*C,Lambda,T,K,m,d,reg_param);
       % [gamma]=SPACL_dim_entropy_EvaluateGamma(X,pi,C,Lambda,T,K,m,d,reg_param,gamma,W);
      %[L_3]=SPACL_dim_entropy_L(X,pi,C,Lambda,gamma,T,d,m,  reg_param, eps_C,W);
      % L_3
       [Lambda]=SPACL_EvaluateLambdaRegularize(pi,gamma,m,K);
      %[L_3]=SPACL_dim_entropy_L(X,pi,C,Lambda,gamma,T,d,m,  reg_param, eps_C,W);
      % L_3
         %[gamma,L3]=EvaluateGammaParallel_Classify(X,pi,C,P,T,d,K,gamma,xt2,pit2,reg_param);
       [C]=SPACL_EvaluateCRegularize_spgqp(X_W,gamma,K,eps_Creg,W_m,d);
       %[L_3]=SPACL_dim_entropy_L(X,pi,C,Lambda,gamma,T,d,m,  reg_param, eps_C,W);
       %L_3
       %[L_3]=SPACL_dim_entropy_L_spgqp(X,pi,C,Lambda,gamma,T,d,m,  reg_param, eps_C,W,K,eps_Creg);
       %L_3
       [W]=SPACL_dim_entropy_EvaluateWRegularize_spgqp(X,gamma,C,d,T,W,eps_C,0);
       %[W]=SPACL_dim_entropy_EvaluateWRegularize(X,gamma,C,d,T,W,eps_C);
       [L_3]=SPACL_dim_entropy_L_spgqp(X,pi,C,Lambda,gamma,T,d,m,  reg_param, eps_C,W,K,eps_Creg);
       %[L_3]=SPACL_dim_entropy_L(X,pi,C,Lambda,gamma,T,d,m,  reg_param, eps_C,W,K,eps_Creg);
       %L_3
        L=[L L_3];
        if i>1 
        delta_L=(L(i-1)-L(i));
        end
%         Ls=0;
%         for i=1:d
%             for k1=1:K
%                 for k2=1:K
%                     Ls=Ls+abs(C(i,k1)-C(i,k2));
%                 end
%             end
%         end
        
        i=i+1;
        figure(11); plot(W,':o');pause(0.05)
        figure(12);plot(L);pause(0.05)
                %[gamma1,LLL2]=EvaluateGammaParallel_Lukas_vect(diag(sqrt(W))*X,diag(sqrt(W))*C,T,d,K,[],xt2_valid);
%                [gamma2]=SPACL_EvaluateGamma_valid(diag(sqrt(W))*X,diag(sqrt(W))*C,T,K);
%        Xx=diag(sqrt(W))*X;
%        pis=Lambda*gamma2;figure(13);clf;plot(Xx(1,round(pis(1,:))==1),Xx(2,round(pis(1,:))==1),'r.');hold on;plot(Xx(1,round(pis(1,:))==0),Xx(2,round(pis(1,:))==0),'bx');plot(C(1,:),C(2,:),'ko','LineWidth',3,'MarkerSize',10)

    end
    out.time=toc;
    T_valid=size(X_valid,2);
    %gamma1{1}=gamma;pi1{1}=pi;
    P=Lambda;%lambdasolver_quadprog_Classify(gamma1,pi1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [gamma1,LLL2]=EvaluateGammaParallel_Lukas_vect(diag(sqrt(W))*X,diag(sqrt(W))*C,T,d,K,[],xt2_valid);
    [L_pred_Markov_train1] = AUC_of_Prediction(gamma1,P,pi,0,flag_AUC);
    L_pred_Markov_train1=L_pred_Markov_train1/(T*m);
    
    [gamma2]=SPACL_EvaluateGamma_valid(diag(sqrt(W))*X,diag(sqrt(W))*C,T,K);
    [L_pred_Markov_train2] = AUC_of_Prediction(gamma2,P,pi,0,flag_AUC);
    L_pred_Markov_train2=L_pred_Markov_train2/(T*m);
     out.N_params=prod(size(C))+prod(size(Lambda))-size(Lambda,2);

%    [gamma_valid,LLL2]=EvaluateGammaParallel(X_valid,C,T_valid,d,K,[],xt2_valid);
    %[gamma_valid]=SPACL_EvaluateGamma_valid(diag(sqrt(W))*X_valid,diag(sqrt(W))*C,T_valid,K);
    if L_pred_Markov_train1< L_pred_Markov_train2
        [gamma_valid,LLL2]=EvaluateGammaParallel_Lukas_vect(diag(sqrt(W))*X_valid,diag(sqrt(W))*C,T_valid,d,K,[],xt2_valid);
    else
        [gamma_valid]=SPACL_EvaluateGamma_valid(diag(sqrt(W))*X_valid,diag(sqrt(W))*C,T_valid,K);
    end
    gamma_valid=real(gamma_valid);
    [L_pred_Markov] = AUC_of_Prediction(gamma_valid,P,pi_valid,0,flag_AUC);
    L_pred_Markov=L_pred_Markov/(T_valid*m);
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

   [L_pred_valid] = AUC_of_Prediction(real(gamma_valid),net,pi_valid,1,flag_AUC);
    L_pred_valid=L_pred_valid/(T_valid*m);
    L_discr_valid=0;
    for t=1:T_valid
        dist=X_valid(:,t)-C*gamma_valid(:,t);
        L_discr_valid=L_discr_valid+dist'*dist;
    end
    L_discr_valid=L_discr_valid/(T_valid*d);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    out.L=L_3;
    out.L_discr_train=L_discr_train;
    out.L_discr_valid=L_discr_valid;
    out.L_pred_valid=L_pred_valid;
    out.L_pred_Markov=L_pred_Markov;
    out.P=P;
    out.W=W;
    out.L_pred_test=L_pred_test;
    %out.norm_C=norm_C;
    out.gamma=gamma;
    out.gamma_valid=gamma_valid;
    out.C=C;
    out.net=net;
    out.reg_param=reg_param;
end

