function [out_opd] = ConvMeansParallelRegularize_Markov(X,pi,K,N_anneal,gamma_init,reg_param,X_valid,pi_valid,N_neurons,flag)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[d,T]=size(X);d=d+size(pi,1);
%xt2=xt2/(T*d);

kkk=1;
for n=1:N_anneal
    gamma=rand(K,T);
    for t=1:T
        gamma(:,t)=gamma(:,t)./sum(gamma(:,t));
    end
    if n==1
        gamma=gamma_init;
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
        xt2=0;%pit2=0;
        xt2_valid=0;
        for t=1:T
            dd=[pi(:,t);reg_param(1,e)*X(:,t)];
            xt2=xt2+dd'*dd;
            %pit2=pit2+pi(:,t)'*pi(:,t);
        end
        for t=1:size(X_valid,2)
            dd=[pi_valid(:,t);reg_param(1,e)*X_valid(:,t)];
            xt2_valid=xt2_valid+dd'*dd;
        end

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
        %in{kkk}.pit2=pit2;
        in{kkk}.reg_param=reg_param(:,e);
        in{kkk}.e=e;
        in{kkk}.n=n;
        in{kkk}.flag=flag;
        in{kkk}.N_anneal=N_anneal;
        in{kkk}.N_neurons=N_neurons;
        kkk=kkk+1;
    end
end
parfor kkk=1:numel(in)
    [out{kkk}]=KHullsAlgorithmReplica_Markov(in{kkk});
end
for kkk=1:numel(in)
    n=in{kkk}.n;
    e=in{kkk}.e;
    L_discr_full(n,e)=out{kkk}.L_discr_train;
    %L_discr_full(n,e)=out{kkk}.L_pred_Markov;
    L(n,e)=out{kkk}.L_pred_test;
    %L_pred(n,e)=out{kkk}.L_pred;
    KKK(n,e)=kkk;
end
for e=1:size(L,2)
    [lll(e),ii]=min(L_discr_full(:,e));
    %L_fin(e)=out{KKK(ii,e)}.L;
    L_fin(e)=out{KKK(ii,e)}.L;
    L_pred_valid(e)=out{KKK(ii,e)}.L_pred_valid;
    L_pred_test(e)=out{KKK(ii,e)}.L_pred_test;
    L_discr_valid(e)=out{KKK(ii,e)}.L_discr_valid;
    L_discr_train(e)=out{KKK(ii,e)}.L_discr_train;
    L_pred_Markov(e)=out{KKK(ii,e)}.L_pred_Markov;
    L_pred_Markov_train(e)=out{KKK(ii,e)}.L_pred_Markov_train;
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
out_opd.L_discr_valid_full=L_discr_valid;
out_opd.L_pred_valid=L_pred_valid(e);
out_opd.L_pred_Markov=L_pred_Markov(e);
out_opd.L_pred_Markov_valid=L_pred_Markov;
out_opd.L_pred_Markov_train=L_pred_Markov_train;
out_opd.P=P(:,:,e);
out_opd.L_discr_train=L_discr_train;
out_opd.norm_C=norm_C;
out_opd.reg_param=reg_param(e);
end

function [out]=KHullsAlgorithmReplica_Markov(in)
    N_neurons=in.N_neurons;
    gamma=in.gamma;
    X_valid=in.X_valid;
    pi_valid=in.pi_valid;
    flag=in.flag;
    T=in.T;
    K=in.K;
    N_anneal=in.N_anneal;
    pi=in.pi;
    m=size(pi,1);
    d=in.d;
    xt2=in.xt2;
    xt2_valid=in.xt2_valid;
    %pit2=in.pit2;
    reg_param=in.reg_param;
    X=[in.pi;reg_param(1)*in.X];
  
    i=1;
    delta_L=1e10;eps=1e-8;
    MaxIter=1000;
    L=[];
    H2=zeros(d*K);
    RegularizationMatrix=zeros(d*K);
    for n_d=1:d
        for k1=1:K
            for k2=1:K
                RegularizationMatrix((k1-1)*d+n_d,(k2-1)*d+n_d)=1;
            end
        end
    end
    
        %for i1=1:K
    H2(:,:)=1/(d^2*K)*(diag(sum(RegularizationMatrix,1))+diag(sum(RegularizationMatrix,2))-2*RegularizationMatrix);
    while and(delta_L>eps,i<=MaxIter)
        if flag
            [C,~,norm_C]=EvaluateCRegularize_Markov(X,gamma,T,d,K,xt2,H2,reg_param(2));
        else
            [C,~,~,norm_C]=EvaluateCRegularize(X,gamma,T,d,K,reg_param(2),xt2);
        end
        %gamma1{1}=gamma;pi1{1}=pi; 
        %P=lambdasolver_quadprog_Classify(gamma1,pi1);
        [gamma,L2]=EvaluateGammaParallel_Lukas_vect(X,C,T,d,K,gamma,xt2);
        Lfin=L2;
        L2=L2+reg_param(2)*norm_C; 
        %[gamma,L3]=EvaluateGammaParallel_Classify(X,pi,C,P,T,d,K,gamma,xt2,pit2,reg_param);
        L=[L L2];
        if i>1
        delta_L=(L(i-1)-L(i));
        end
        i=i+1;
        %gamma_full(:,:,i)=gamma;
    end
    T_valid=size(X_valid,2);
    gamma1{1}=gamma;pi1{1}=pi;
    P=C(1:m,:);%lambdasolver_quadprog_Classify(gamma1,pi1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%    [gamma_valid,LLL2]=EvaluateGammaParallel(X_valid,C,T_valid,d,K,[],xt2_valid);
    [gamma_valid,LLL2]=EvaluateGammaParallel_Lukas_vect(reg_param(1)*X_valid,C((m+1):size(C,1),:),T_valid,d,K,[],xt2_valid);
    L_pred_Markov_train=0;
    for t=1:T
        L_pred_Markov_train=L_pred_Markov_train+norm(pi(:,t)-P*gamma(:,t),2)^2;
    end
    L_pred_Markov_train=L_pred_Markov_train/(T*m);
    L_pred_Markov=0;
    for t=1:T_valid
        L_pred_Markov=L_pred_Markov+norm(pi_valid(:,t)-P*gamma_valid(:,t),2)^2;
    end
    L_pred_Markov=L_pred_Markov/(T_valid*m);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for n=1:N_anneal
        net = feedforwardnet(N_neurons);
        net.trainParam.showWindow = 0;
        net = train(net,gamma,pi,'useParallel','no');
        err=0;
        for t=1:T
            err=err+norm(pi(:,t)-net(gamma(:,t)),2)^2;
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
  L_pred_test=0;
    for t=1:T
        L_pred_test=L_pred_test+norm(pi(:,t)-net(gamma(:,t)),2)^2;
    end
    L_pred_test=L_pred_test/(T*m);
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    L_pred_valid=0;
    for t=1:T_valid
        L_pred_valid=L_pred_valid+norm(pi_valid(:,t)-net(gamma_valid(:,t)),2)^2;
    end
    L_pred_valid=L_pred_valid/(T_valid*m);
    L_discr_train=0;
    for t=1:T
        dist=X((m+1):size(C,1),t)./reg_param(1)-C((m+1):size(C,1),:)./reg_param(1)*gamma(:,t);
        L_discr_train=L_discr_train+dist'*dist;
    end
    L_discr_train=L_discr_train/(T*d);
    L_discr_valid=0;
    for t=1:T_valid
        dist=X_valid(:,t)-C((m+1):size(C,1),:)./reg_param(1)*gamma_valid(:,t);
        L_discr_valid=L_discr_valid+dist'*dist;
    end
    L_discr_valid=L_discr_valid/(T_valid*d);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    out.L=Lfin;
    out.L_discr_train=L_discr_train;
    out.L_discr_valid=L_discr_valid;
    out.L_pred_valid=L_pred_valid;
    out.L_pred_Markov=L_pred_Markov;
    out.L_pred_Markov_train=L_pred_Markov_train;
    out.P=P;
    out.L_pred_test=L_pred_test;
    out.norm_C=norm_C;
    out.gamma=gamma;
    out.gamma_valid=gamma_valid;
    out.C=C;
    out.net=net;
    out.reg_param=reg_param;
end
