function [out_opd] = SPACL_kmeans_dim_entropy_analytic_simple_new(X,pi,K,N_anneal,out_init,reg_param,flag_AUC)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[d,T]=size(X);m=size(pi,1);
xt2=0;pit2=0;xt2_valid=0;
for t=1:T
    xt2=xt2+X(:,t)'*X(:,t);
    pit2=pit2+pi(:,t)'*pi(:,t);
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
    W=ones(1,d)./d;
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
    time(n,e)=out{kkk}.time;
    %N_params(n,e)=out{kkk}.N_params;
end
for e=1:size(L_pred,2)
    %[lll(e),ii]=min(L_discr_full(:,e));
    [lll(e),ii]=min(L_pred(:,e));
    %L_fin(e)=out{KKK(ii,e)}.L;
    L_fin(e)=out{KKK(ii,e)}.L;
    L_discr_valid(e)=out{KKK(ii,e)}.L_discr_valid;
    L_discr_train(e)=out{KKK(ii,e)}.L_discr_train;
    L_pred_Markov(e)=out{KKK(ii,e)}.L_pred_Markov;
    N_Markov(e)=out{KKK(ii,e)}.N_params;
    gamma_fin(:,:,e)=out{KKK(ii,e)}.gamma;
    gamma_valid_fin(:,:,e)=out{KKK(ii,e)}.gamma_valid;
    P(:,:,e)=out{KKK(ii,e)}.P;
    P_valid(:,:,e)=out{KKK(ii,e)}.P_valid;
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
out_opd.L=L_fin(e);
out_opd.L_pred_valid_Markov=L_pred_Markov(e);
out_opd.L_pred_valid_Markov_full=L_pred;
out_opd.P=P(:,:,e);
out_opd.P_valid=P_valid(:,:,e);
out_opd.L_discr_train=L_discr_train;
%out_opd.norm_C=norm_C;
out_opd.reg_param_W=reg_param(1,e);
out_opd.reg_param_CL=reg_param(2,e);
out_opd.W=W_fin(:,e);
out_opd.time_Markov=mean(mean(time));
out_opd.N_params_Markov=N_Markov(e);
end

function [out]=SPACL_Replica(in)
    eps_C=in.reg_param(1);
    X=in.X;
    gamma=in.gamma;
    C=in.C;
    W=in.W;
    T=in.T;
    K=in.K;
    pi=in.pi;
    flag_AUC=in.flag_AUC;
    Lambda=in.Lambda;
    m=size(pi,1);
    d=in.d;
    reg_param=in.reg_param(2);
    i=1;
    delta_L=1e10;eps=1e-11;
    eps_Creg=0;
    MaxIter=300;
    L=[];
    tic;
   
    while and(delta_L>eps,i<=MaxIter)
      %[L_3]=SPACL_dim_entropy_L(X,pi,C,Lambda,gamma,T,d,m,  reg_param, eps_C,W);
      % L_3
      if i==1
          W_m=sqrt(W)';
          X_W=bsxfun(@times,X,W_m);
      end
      [C]=SPACL_EvaluateCRegularize_analytic(X,gamma,K,d,T);
      C_W=bsxfun(@times,C,W_m);     
      [gamma]=SPACL_EvaluateGamma(X_W,pi,C_W,Lambda,T,K,m,d,reg_param);
      %[gamma]=SPACL_EvaluateGamma(diag(sqrt(W))*X,pi,diag(sqrt(W))*C,Lambda,T,K,m,d,reg_param);
       % [gamma]=SPACL_dim_entropy_EvaluateGamma(X,pi,C,Lambda,T,K,m,d,reg_param,gamma,W);
      %[L_3]=SPACL_dim_entropy_L(X,pi,C,Lambda,gamma,T,d,m,  reg_param, eps_C,W);
      % L_3
       [Lambda]=SPACL_EvaluateLambdaRegularize(pi,gamma,m,K);
      %[L_3]=SPACL_dim_entropy_L(X,pi,C,Lambda,gamma,T,d,m,  reg_param, eps_C,W);
      % L_3
         %[gamma,L3]=EvaluateGammaParallel_Classify(X,pi,C,P,T,d,K,gamma,xt2,pit2,reg_param);
       %[L_3]=SPACL_dim_entropy_L(X,pi,C,Lambda,gamma,T,d,m,  reg_param, eps_C,W);
       %L_3
       %[W]=SPACL_dim_entropy_EvaluateWRegularize_v2(X,gamma,C,d,T,W,eps_C);
       [W]=SPACL_dim_entropy_EvaluateWRegularize_v3(X,gamma,C,d,T,W,eps_C);
       %[W]=SPACL_dim_entropy_EvaluateWRegularize(X,gamma,C,d,T,W,eps_C);
       W_m=sqrt(W)';
       X_W=bsxfun(@times,X,W_m);     
       C_W=bsxfun(@times,C,W_m);     
       [L_3]=SPACL_dim_entropy_L(X_W,pi,C_W,Lambda,gamma,T,d,m,  reg_param, eps_C,W,K,eps_Creg);
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
        %figure(11); plot(W,':o');pause(0.05)
        %figure(12);plot(L);pause(0.05)
                %[gamma1,LLL2]=EvaluateGammaParallel_Lukas_vect(diag(sqrt(W))*X,diag(sqrt(W))*C,T,d,K,[],xt2_valid);
%                [gamma2]=SPACL_EvaluateGamma_valid(diag(sqrt(W))*X,diag(sqrt(W))*C,T,K);
%        Xx=diag(sqrt(W))*X;
%        pis=Lambda*gamma2;figure(13);clf;plot(Xx(1,round(pis(1,:))==1),Xx(2,round(pis(1,:))==1),'r.');hold on;plot(Xx(1,round(pis(1,:))==0),Xx(2,round(pis(1,:))==0),'bx');plot(C(1,:),C(2,:),'ko','LineWidth',3,'MarkerSize',10)

    end
    out.time=toc;
    %gamma1{1}=gamma;pi1{1}=pi;
    P=Lambda;%lambdasolver_quadprog_Classify(gamma1,pi1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    [gamma2]=SPACL_EvaluateGamma_valid(diag(sqrt(W))*X,C_W,T,K);
    %[L_pred_Markov_train2] = AUC_of_Prediction(gamma2,P,pi,0,flag_AUC);
    K_actual=length(find(sum(gamma')>1e-7));
    out.N_params=d*(K_actual+1)+(m-1)*K_actual;
%    out.N_params=prod(size(C))+prod(size(Lambda))-size(Lambda,2);
    %[outliers,~,~,cc]=isoutlier(W,'grubbs','ThresholdFactor',0.01);
    %ii = find(and(outliers==1,outliers>cc));
    %if length(ii)==0
    %   ii=1:d; 
    %end
    %dd=length(ii);
    %out.N_params=dd+dd*K+(m-1)*K;%d*(K+1);%length(W)*(K+1);%length(find(W>1e-5))*(K+1);

%    [gamma_valid,LLL2]=EvaluateGammaParallel(X_valid,C,T_valid,d,K,[],xt2_valid);
    %[gamma_valid]=SPACL_EvaluateGamma_valid(diag(sqrt(W))*X_valid,diag(sqrt(W))*C,T_valid,K);
    [gamma_valid]=SPACL_EvaluateGamma_valid(diag(sqrt(W))*X,C_W,T,K);
    gamma_valid=real(gamma_valid);
    [Lambda_valid]=SPACL_EvaluateLambdaRegularize(pi,gamma_valid,m,K);
    [L_pred_Markov] = AUC_of_Prediction(gamma_valid,P,pi,0,flag_AUC);
    L_pred_Markov=L_pred_Markov/(T*m);
    %gamma_new=Lambda_valid*gamma_valid;
    % [~,ind]=max(gamma_new);
    % [~,ind_true]=max(pi);

    %[~,L_pred_Markov] = F1_and_Accuracy(ind_true,ind);
    %L_pred_Markov=-L_pred_Markov;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      L_discr_train=0;
    for t=1:T
        dist=X(:,t)-C*gamma(:,t);
        L_discr_train=L_discr_train+dist'*dist;
    end
    L_discr_train=L_discr_train/(T*d);
    L_discr_valid=0;
    for t=1:T
        dist=X(:,t)-C*gamma_valid(:,t);
        L_discr_valid=L_discr_valid+dist'*dist;
    end
    L_discr_valid=L_discr_valid/(T*d);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    out.L=L_3;
    out.L_discr_train=L_discr_train;
    out.L_discr_valid=L_discr_valid;
    out.L_pred_Markov=L_pred_Markov;
    out.P=P;
    out.P_valid=Lambda_valid;
    out.W=W;
    %out.norm_C=norm_C;
    out.gamma=gamma;
    out.gamma_valid=gamma_valid;
    out.C=C;
    out.reg_param=reg_param;
end

