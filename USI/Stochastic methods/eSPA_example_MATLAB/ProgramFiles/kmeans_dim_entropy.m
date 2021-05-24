function [out] = kmeans_dim_entropy(X,K,N_anneal,reg_param)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[d,T]=size(X);
%xt2=xt2/(T*d);

kkk=1;
for n=1:N_anneal
    for e=1:length(reg_param)
        in{kkk}.gamma=sparse(randi(K,1,T),1:T,1);
        in{kkk}.W=ones(1,d)./d;
        in{kkk}.X=X;
        in{kkk}.reg_param=reg_param(e);
        in{kkk}.K=K;
        in{kkk}.T=T;
        in{kkk}.d=d;
        in{kkk}.n=n;
        in{kkk}.e=e;
        kkk=kkk+1;
    end
end
parfor i=1:kkk-1
    outs{i}=SPACL_Replica(in{i});
end
for kkk=1:numel(in)
    n=in{kkk}.n;
    e=in{kkk}.e;
    N_params(n,e)=outs{kkk}.N_params;
    %L(n,e)=out{kkk}.L_fin;
    L(n,e)=outs{kkk}.L;
    AIC(n,e)=outs{kkk}.AIC;
    BIC(n,e)=outs{kkk}.BIC;
    KKK(n,e)=kkk;
    time(n,e)=outs{kkk}.time;
    N_params(n,e)=outs{kkk}.N_params;
end
for e=1:size(L,2)
    %[lll(e),ii]=min(L_discr_full(:,e));
    [lll(e),ii]=min(L(:,e));
    %L_fin(e)=out{KKK(ii,e)}.L;
    out.L(e)=L(ii,e);
    out.AIC(e)=AIC(ii,e);
    out.BIC(e)=BIC(ii,e);
    out.N_params(e)=N_params(ii,e);
    out.gamma{e}=outs{KKK(ii,e)}.gamma;
    out.C{e}=outs{KKK(ii,e)}.C;
   out.W{e}=outs{KKK(ii,e)}.W;
    %norm_C(e,:)=out{KKK(ii,e)}.norm_C;
end

end
function [out]=SPACL_Replica(in)
    eps_C=in.reg_param;
    X=in.X;
    gamma=in.gamma;
    W=in.W;
    T=in.T;
    K=in.K;
    d=in.d;
    i=1;
    delta_L=1e10;eps=1e-8;
    eps_Creg=1e-10;
    MaxIter=100;
    L=[];
    tic;
   
    while and(delta_L>eps,i<=MaxIter)
      %[L_3]=SPACL_dim_entropy_L(X,pi,C,Lambda,gamma,T,d,m,  reg_param, eps_C,W);
      % L_3
      if i==1
      W_m=sqrt(W)';
      X_W=bsxfun(@times,X,W_m);
      end
      %[gamma]=SPACL_EvaluateGamma(diag(sqrt(W))*X,pi,diag(sqrt(W))*C,Lambda,T,K,m,d,reg_param);
       % [gamma]=SPACL_dim_entropy_EvaluateGamma(X,pi,C,Lambda,T,K,m,d,reg_param,gamma,W);
      %[L_3]=SPACL_dim_entropy_L(X,pi,C,Lambda,gamma,T,d,m,  reg_param, eps_C,W);
      % L_3
      %[L_3]=SPACL_dim_entropy_L(X,pi,C,Lambda,gamma,T,d,m,  reg_param, eps_C,W);
      % L_3
         %[gamma,L3]=EvaluateGammaParallel_Classify(X,pi,C,P,T,d,K,gamma,xt2,pit2,reg_param);
       [C]=SPACL_EvaluateCRegularize(X_W,gamma,K,eps_Creg);
        %[L_3]=kmeans_dim_entropy_L(X_W,C,gamma,T,d, eps_C,W,K,eps_Creg)
       [gamma]=kmeans_entropy_EvaluateGamma(X_W,C,T,d,K);
       %[L_3]=SPACL_dim_entropy_L(X,pi,C,Lambda,gamma,T,d,m,  reg_param, eps_C,W);
       %L_3
        %[L_3]=kmeans_dim_entropy_L(X_W,C,gamma,T,d, eps_C,W,K,eps_Creg)
       [W]=SPACL_dim_entropy_EvaluateWRegularize_v2(X,gamma,C,d,T,W,eps_C);
      %[W]=SPACL_dim_entropy_EvaluateWRegularize(X,gamma,C,d,T,W,eps_C);
        W_m=sqrt(W)';
        X_W=bsxfun(@times,X,W_m);
       [L_3,res]=kmeans_dim_entropy_L(X_W,C,gamma,T,d, eps_C,W,K,eps_Creg);
       %[L_3]=SPACL_dim_entropy_L(X,pi,C,Lambda,gamma,T,d,m,  reg_param, eps_C,W,K,eps_Creg);
       %L_3
        L=[L L_3];
        if i>1 
        delta_L=(L(i-1)-L(i));
        end
        
        i=i+1;
        %figure(11); plot(W,':o');pause(0.05)
        %figure(12);plot(L);pause(0.05)
                %[gamma1,LLL2]=EvaluateGammaParallel_Lukas_vect(diag(sqrt(W))*X,diag(sqrt(W))*C,T,d,K,[],xt2_valid);
%                [gamma2]=SPACL_EvaluateGamma_valid(diag(sqrt(W))*X,diag(sqrt(W))*C,T,K);
%        Xx=diag(sqrt(W))*X;
%        pis=Lambda*gamma2;figure(13);clf;plot(Xx(1,round(pis(1,:))==1),Xx(2,round(pis(1,:))==1),'r.');hold on;plot(Xx(1,round(pis(1,:))==0),Xx(2,round(pis(1,:))==0),'bx');plot(C(1,:),C(2,:),'ko','LineWidth',3,'MarkerSize',10)

    end
    out.time=toc;
    out.L=L_3;
    [outliers,~,~,cc]=isoutlier(W,'mean');
    ii = find(and(outliers==1,outliers>cc));
    if length(ii)==0
       ii=1:d; 
    end
    LogLik = LogLikGMM_OfReiduals(X(ii,:),gamma);
    dd=length(ii);
    out.N_params=dd+(dd+dd^2)*K;%d*(K+1);%length(W)*(K+1);%length(find(W>1e-5))*(K+1);
    out.AIC=-2*LogLik+2*out.N_params;
    out.BIC=-2*LogLik+log(T)*out.N_params;
    out.gamma=gamma;
    out.W=W;
    out.C=C;
    out.red_dim=ii;
end

