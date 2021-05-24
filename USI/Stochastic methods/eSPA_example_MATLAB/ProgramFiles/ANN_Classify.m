function [out]= ANN_Classify(X,pi,K, X_valid, pi_valid,N_anneal,N_neurons)
[n,T]=size(X);m=size(pi,1);
[out.idx_fin,out.C_fin,L_fin]=kmeans(X',K,'Replicates',N_anneal,'MaxIter',1000);
out.gamma=zeros(K,T);
for ttt=1:T
    out.gamma(out.idx_fin(ttt),ttt)=1;
end
out.C_fin=out.C_fin';
out.L_fin=0;
for t=1:T
    err=X(:,t)-out.C_fin(:,out.idx_fin(t));
    out.L_fin=out.L_fin+err'*err;
end
out.L_fin=out.L_fin/T/n;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T_valid=size(X_valid,2);
out.gamma_valid=zeros(K,T_valid);
for t=1:T_valid
    for k=1:K
        vvv=X_valid(:,t)-out.C_fin(:,k);
        dist(k)=vvv'*vvv;
    end
    [~,ii]=min(dist);
    out.gamma_valid(ii,t)=1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for n=1:N_anneal
    net = patternnet(N_neurons);
    net.trainParam.showWindow = 0;
    net = train(net,out.gamma,pi,'useParallel','yes');
    err=0;
    for t=1:T
        err=err+KLdivergence(pi(:,t),net(out.gamma(:,t)));
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
%view(net)
out.err_valid_pred = 0;
for t=1:T_valid
    out.err_valid_pred = out.err_valid_pred+KLdivergence(pi_valid(:,t),net_final(out.gamma_valid(:,t)));
end
out.err_valid_pred=out.err_valid_pred/(m*T_valid);
out.net=net_final;

