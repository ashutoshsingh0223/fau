function [out_opd] = ConvMeansParallelRegularize_Classify_v2(X,pi,reg_param,X_valid,pi_valid,K,N_anneal)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[d,T]=size(X);
xt2=0;pit2=0;
for t=1:T
    xt2=xt2+X(:,t)'*X(:,t);
end
kkk=1;

for n=1:N_anneal
     for e=1:length(reg_param)
        in{kkk}.X=X;
        in{kkk}.pi=pi;
        in{kkk}.X_valid=X_valid;
        in{kkk}.pi_valid=pi_valid;
        in{kkk}.T=T;
        in{kkk}.K=K;
        in{kkk}.d=d;
        in{kkk}.xt2=xt2;
        in{kkk}.pit2=pit2;
        in{kkk}.reg_param=reg_param(e);
        in{kkk}.e=e;
        in{kkk}.n=n;
        kkk=kkk+1;
    end
end
for kkk=1:numel(in)
    [out{kkk}]=KHullsAlgorithmReplica_Classify_v2(in{kkk});
end
for kkk=1:numel(in)
    n=in{kkk}.n;
    e=in{kkk}.e;
    L(n,e)=out{kkk}.L;
    %L_pred(n,e)=out{kkk}.L_pred;
    KKK(n,e)=kkk;
end
for e=1:size(L,2)
    ii=1;L_fin(e)=L(1,e);
    %Lfin(e)=out{KKK(ii,e)}.L;
    L_pred_valid(e)=out{KKK(ii,e)}.L_pred_valid;
    L_discr_valid(e)=out{KKK(ii,e)}.L_discr_valid;
    L_pred_train(e)=out{KKK(ii,e)}.L_pred_train;
    L_discr_train(e)=out{KKK(ii,e)}.L_discr_train;
    gamma_valid_fin(:,:,e)=out{KKK(ii,e)}.gamma_valid;
    C_fin(:,:,e)=out{KKK(ii,e)}.C;
    Lambda_fin(:,:,e)=out{KKK(ii,e)}.Lambda;
    norm_C(e)=out{KKK(ii,e)}.norm_C;
end
out_opd.gamma_valid=gamma_valid_fin;
out_opd.C=C_fin;
out_opd.Lambda=Lambda_fin;
out_opd.L=L_fin;
out_opd.L_discr_valid=L_discr_valid;
out_opd.L_pred_valid=L_pred_valid;
out_opd.L_discr_train=L_discr_train;
out_opd.L_pred_train=L_pred_train;
out_opd.norm_C=norm_C;
end

function [out]=KHullsAlgorithmReplica_Classify_v2(in)
    X=in.X;
    X_valid=in.X_valid;
    pi_valid=in.pi_valid;
    T=in.T;
    K=in.K;
    pi=in.pi;
    m=size(pi,1);
    d=in.d;
    Lambda=rand(K,m);
    for j=1:m
       Lambda(:,j)=Lambda(:,j)/sum(Lambda(:,j)); 
    end
    xt2=in.xt2;
    reg_param=in.reg_param;
    T_valid=size(X_valid,2);
    i=1;
    delta_L=1e10;eps=1e-7;
    MaxIter=100;
    L=[];
    while and(delta_L>eps,i<=MaxIter)
        [C,L1,L0,norm_C]=EvaluateCRegularize(X,Lambda*pi,T,d,K,reg_param,xt2);
        [Lambda, timers, lm, output] = lambdasolver_label_quadprog( X, pi, C );
        L=[L L1];
        if i>1
            delta_L=(L(i-1)-L(i));
        end
        i=i+1;
    end
    
    [gamma_valid,L3]=EvaluateLabelingParallel(X_valid,C*Lambda,T_valid,d,m,xt2);
    [gamma,~]=EvaluateLabelingParallel(X,C*Lambda,T,d,m,xt2);
   %T_valid=size(X_valid,2);
    [m,T_valid]=size(pi_valid);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    L_pred=0;
    for t=1:T_valid
        L_pred=L_pred+KLdivergence(pi_valid(:,t),gamma_valid(:,t));
    end
    L_pred=L_pred/(T_valid*m);
    L_discr=0;
    for t=1:T_valid
        dist=X_valid(:,t)-C*Lambda*gamma_valid(:,t);
        L_discr=L_discr+dist'*dist;
    end
    L_discr=L_discr/(T_valid*d);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    L_pred_train=0;
    for t=1:T
        L_pred_train=L_pred_train+KLdivergence(pi(:,t),gamma(:,t));
    end
    L_pred_train=L_pred_train/(T*m);
    L_discr_train=0;
    for t=1:T
        dist=X(:,t)-C*Lambda*gamma(:,t);
        L_discr_train=L_discr_train+dist'*dist;
    end
    L_discr_train=L_discr_train/(T*d);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   out.L=L1;
    out.L_discr_valid=L_discr;
    out.L_pred_valid=L_pred;
    out.L_discr_train=L_discr_train;
    out.L_pred_train=L_pred_train;
    out.norm_C=norm_C;
    out.Lambda=Lambda;
    out.gamma_valid=gamma_valid;
    out.C=C;
end

