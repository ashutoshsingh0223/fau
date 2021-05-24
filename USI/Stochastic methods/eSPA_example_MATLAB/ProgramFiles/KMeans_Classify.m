function [out_kmeans]= KMeans_Classify(X,pi,K, X_valid, pi_valid,N_anneal,flag_AUC)   
    [d,T]=size(X);m=size(pi,1);
    tic;
    [out_kmeans.idx_fin,out_kmeans.C_fin,L_fin]=kmeans(X',K,'Replicates',N_anneal,'MaxIter',100000,'Options',statset('TolFun',1e-8));
    out_kmeans.gamma=zeros(K,T);
    for ttt=1:T
       out_kmeans.gamma(out_kmeans.idx_fin(ttt),ttt)=1;
    end
    out_kmeans.C_fin=out_kmeans.C_fin';
    out_kmeans.L_fin=0;
    for t=1:T
        err=X(:,t)-out_kmeans.C_fin(:,out_kmeans.idx_fin(t));
        out_kmeans.L_fin=out_kmeans.L_fin+err'*err;
    end
    out_kmeans.L_fin=out_kmeans.L_fin/T/d;
    time_kmeans=toc;

    gam{1}=out_kmeans.gamma;p{1}=pi;
    tic;
    N_Params_Kmeans=prod(size( out_kmeans.C_fin));
    P=lambdasolver_quadprog_Classify(gam,p);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   [out_kmeans.err_pred] = AUC_of_Prediction(out_kmeans.gamma,P,pi,0,flag_AUC);
     out_kmeans.err_pred= out_kmeans.err_pred/(m*T);
     
%        err_disc=0;
%    for t=1:T
%       dev_disc=X(:,t)-C*out_kmeans.gamma;
%       err_disc=err_disc+dev_disc'*dev_disc;
%    end
%    err_disc=err_disc/(d*T);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    T_valid=size(X_valid,2);
    out_kmeans.gamma_valid=zeros(K,T_valid);
    out_kmeans.err_valid_discr=0;
    out_kmeans.err_valid_pred=0;
    for t=1:T_valid
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
   [out_kmeans.L_pred_valid_Markov] = AUC_of_Prediction(out_kmeans.gamma_valid,P,pi_valid,0,flag_AUC);
   out_kmeans.time_Markov=time_kmeans+toc;
   out_kmeans.N_params_Markov=N_Params_Kmeans+size(P,2)*(size(P,1)-1);

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
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %net = patternnet(N_neurons);net.trainParam.showWindow = 0;net = train(net,gamma,pi,'useParallel','no');
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        out_kmeans.net{n_neurons}=net_final;
        out_kmeans.N_params(n_neurons)=N_Params_Kmeans+prod(size(net_final.IW{1}))+prod(size(net_final.IW{2}))...
            +prod(size(net_final.LW{1}))+prod(size(net_final.LW{2}))...
            +prod(size(net_final.b{1}))+prod(size(net_final.b{2}));
        L_discr_train(n_neurons)=0;
        for t=1:T
            dist=X(:,t)-out_kmeans.C_fin*out_kmeans.gamma(:,t);
            L_discr_train(n_neurons)=L_discr_train(n_neurons)+dist'*dist;
        end
        L_discr_train(n_neurons)=L_discr_train(n_neurons)/(T*d);
        [L_pred_test(n_neurons)] = AUC_of_Prediction(out_kmeans.gamma,net,pi,1,flag_AUC);
        L_pred_test(n_neurons)=L_pred_test(n_neurons)/(T*m);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        [L_pred_valid(n_neurons)] = AUC_of_Prediction(out_kmeans.gamma_valid,out_kmeans.net{n_neurons},pi_valid,1,flag_AUC);
        L_pred_valid(n_neurons)=L_pred_valid(n_neurons)/(T_valid*m);
        L_discr_valid(n_neurons)=0;
        for t=1:T_valid
            dist=X_valid(:,t)-out_kmeans.C_fin*out_kmeans.gamma_valid(:,t);
            L_discr_valid(n_neurons)=L_discr_valid(n_neurons)+dist'*dist;
        end
        L_discr_valid(n_neurons)=L_discr_valid(n_neurons)/(T_valid*d);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    out_kmeans.L_discr_train=L_discr_train;
    out_kmeans.L_discr_valid=L_discr_valid;
    out_kmeans.L_pred_valid=L_pred_valid;
    out_kmeans.L_pred_test=L_pred_test;
    out_kmeans.time=time_kmeans+mean(time);
    %out_kmeans.net=net;
    out_kmeans.err_valid_discr=out_kmeans.err_valid_discr/(T_valid*n);
    out_kmeans.L_pred_valid_Markov=out_kmeans.L_pred_valid_Markov/(T_valid*m);
    out_kmeans.P=P;