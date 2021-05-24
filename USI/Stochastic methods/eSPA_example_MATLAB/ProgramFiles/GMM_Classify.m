function [out_kmeans]= GMM_Classify(X,pi,K, X_valid, pi_valid,N_anneal,flag_AUC)
[n,T]=size(X);m=size(pi,1);
tic;
GMModel=fitgmdist(X',K,'CovarianceType','diagonal','Replicates',N_anneal,...
    'RegularizationValue',1e-8,...
    'Options',statset('Display','off','MaxIter',15000));
out_kmeans.gamma=GMModel.posterior(X')';
out_kmeans.C_fin=GMModel.mu';
out_kmeans.L_fin=0;
for t=1:T
    x_rec=out_kmeans.C_fin*out_kmeans.gamma(:,t);
    err=X(:,t)-x_rec;
    out_kmeans.L_fin=out_kmeans.L_fin+err'*err;
end
out_kmeans.L_fin=out_kmeans.L_fin/T/n;
time_gmm=toc;

gam{1}=out_kmeans.gamma;p{1}=pi;
tic;
N_Params_gmm=2*prod(size(GMModel.mu));
P=lambdasolver_quadprog_Classify(gam,p);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[out_kmeans.err_pred] = AUC_of_Prediction(out_kmeans.gamma,P,pi,0,flag_AUC);
out_kmeans.err_pred= out_kmeans.err_pred/(m*T);
out_kmeans.time_Markov=time_gmm+toc;
out_kmeans.N_params_Markov=N_Params_gmm+size(P,2)*(size(P,1)-1);
%        err_disc=0;
%    for t=1:T
%       dev_disc=X(:,t)-C*out_kmeans.gamma;
%       err_disc=err_disc+dev_disc'*dev_disc;
%    end
%    err_disc=err_disc/(d*T);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T=size(X_valid,2);
out_kmeans.gamma_valid=zeros(K,T);
out_kmeans.err_valid_discr=0;
out_kmeans.err_valid_pred=0;
for t=1:T
    for k=1:K
        vvv=X_valid(:,t)-out_kmeans.C_fin(:,k);
        dist(k)=vvv'*vvv;
    end
    [~,ii]=min(dist);
    out_kmeans.gamma_valid(ii,t)=1;
    %ppp=pi_valid(:,t)-P*out_kmeans.gamma_valid(:,t);
    %out_kmeans.err_valid_pred=out_kmeans.err_valid_pred+KLdivergence(pi_valid(:,t),P*out_kmeans.gamma_valid(:,t));%ppp'*ppp;
    out_kmeans.err_valid_discr=out_kmeans.err_valid_discr+dist(ii);
end
out_kmeans.err_valid_discr=out_kmeans.err_valid_discr/(T*n);
[out_kmeans.L_pred_valid_Markov] = AUC_of_Prediction(out_kmeans.gamma_valid,P,pi_valid,0,flag_AUC);
out_kmeans.L_pred_valid_Markov=out_kmeans.L_pred_valid_Markov/(T*m);
out_kmeans.P=P;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n,T]=size(X);
T_valid=size(X_valid,2);
N_neurons=[1:10 15 20];
for n_neurons=1:length(N_neurons)
    %N_neurons(n_neurons)
    for n=1:N_anneal
        tic;
        net = patternnet(N_neurons(n_neurons));
        net.trainParam.showWindow = 0;
        net = train(net,out_kmeans.gamma,pi,'useParallel','no');
        time(n,n_neurons)=toc;
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
    out_kmeans.net{n_neurons}=net_final;
    out_kmeans.N_params(n_neurons)=N_Params_gmm+prod(size(net_final.IW{1}))+prod(size(net_final.IW{2}))...
        +prod(size(net_final.LW{1}))+prod(size(net_final.LW{2}))...
        +prod(size(net_final.b{1}))+prod(size(net_final.b{2}));
    
    %view(net)
    [out_kmeans.L_pred_valid(n_neurons)] = AUC_of_Prediction(out_kmeans.gamma_valid,net_final,pi_valid,1,flag_AUC);
    out_kmeans.L_pred_valid(n_neurons)=out_kmeans.L_pred_valid(n_neurons)/(m*T_valid);
    out_kmeans.net{n_neurons}=net_final;
end
    out_kmeans.time=time_gmm+mean(time);


