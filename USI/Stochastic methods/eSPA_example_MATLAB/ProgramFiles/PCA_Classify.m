function [out_kmeans]= PCA_Classify(X,pi,K, X_valid, pi_valid,N_anneal,flag_AUC)
[d,T]=size(X);m=size(pi,1);
tic;
[out_kmeans.gamma,V,mu]=MyPCA(X,min(K,d));
time_pca=toc;
N_Params_pca=prod(size(V))+prod(size(mu));

%     [out_kmeans.idx_fin,out_kmeans.C_fin,L_fin]=kmeans(X',K,'Replicates',N_anneal,'MaxIter',1000);
%     out_kmeans.gamma=zeros(K,T);
%     for ttt=1:T
%        out_kmeans.gamma(out_kmeans.idx_fin(ttt),ttt)=1;
%     end
out_kmeans.C_fin=V;
out_kmeans.L_fin=0;
for t=1:T
    err=X(:,t)-mu-V*out_kmeans.gamma(:,t);
    out_kmeans.L_fin=out_kmeans.L_fin+err'*err;
end
out_kmeans.L_fin=out_kmeans.L_fin/T/d;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_neurons=[1:10 15 20];
for n_neurons=1:length(N_neurons)
    %N_neurons(n_neurons)
    for n=1:N_anneal
        tic;
        net = patternnet(N_neurons(n_neurons));
        net.trainParam.showWindow = 0;
        net = train(net,out_kmeans.gamma,pi,'useParallel','no');
        out_kmeans.time(n,n_neurons)=time_pca+toc;
        err=0;
        for t=1:T
            err=err+KLdivergence(pi(:,t),net(out_kmeans.gamma(:,t)));
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
    out_kmeans.net{n_neurons}=net_final;
    out_kmeans.N_params(n_neurons)=N_Params_pca+prod(size(net_final.IW{1}))+prod(size(net_final.IW{2}))...
        +prod(size(net_final.LW{1}))+prod(size(net_final.LW{2}))...
        +prod(size(net_final.b{1}))+prod(size(net_final.b{2}));
    L_discr_train(n_neurons)=0;
    L_discr_train(n_neurons)=0;
    for t=1:T
        dist=X(:,t)-mu-V*out_kmeans.gamma(:,t);
        L_discr_train(n_neurons)=L_discr_train(n_neurons)+dist'*dist;
    end
    L_discr_train(n_neurons)=L_discr_train(n_neurons)/(T*d);
    [L_pred_test(n_neurons)] = AUC_of_Prediction(out_kmeans.gamma,out_kmeans.net{n_neurons},pi,1,flag_AUC);
    L_pred_test(n_neurons)=L_pred_test(n_neurons)/(T*m);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    T_valid=size(X_valid,2);
    gamma_valid=V'*(X_valid-repmat(mu,1,T_valid));
    
    [L_pred_valid(n_neurons)] = AUC_of_Prediction(gamma_valid,out_kmeans.net{n_neurons},pi_valid,1,flag_AUC);
    L_pred_valid(n_neurons)=L_pred_valid(n_neurons)/(T_valid*m);
    L_discr_valid(n_neurons)=0;
    for t=1:T_valid
        dist=X_valid(:,t)-mu-V*gamma_valid(:,t);
        L_discr_valid(n_neurons)=L_discr_valid(n_neurons)+dist'*dist;
    end
    L_discr_valid(n_neurons)=L_discr_valid(n_neurons)/(T_valid*d);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
    out_kmeans.L_discr_train=L_discr_train;
    out_kmeans.L_discr_valid=L_discr_valid;
    out_kmeans.L_pred_valid=L_pred_valid;
    out_kmeans.L_pred_test=L_pred_test;
    out_kmeans.time=mean(out_kmeans.time);
    out_kmeans.gamma_valid=gamma_valid;
    out_kmeans.V=V;
    %out_kmeans.net=net;

