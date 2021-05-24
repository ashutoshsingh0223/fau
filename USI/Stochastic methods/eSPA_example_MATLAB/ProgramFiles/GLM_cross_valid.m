function [AUClog] = GLM_cross_valid(X,Y,N_ens)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
T=length(Y);
fraction=3/4;
for n_ens=1:N_ens
    n_ens
    %%% Generating a random bisection of the data into the training and
    %%% the validation sets
    ind_perm=randperm(T);
    [~,ind_back]=sort(ind_perm,'ascend');
    X=X(:,ind_perm);
    Y=Y(ind_perm);
    X_train=X(:,1:floor(fraction*T));Y_train=Y(1:floor(fraction*T));
    X_valid=X(:,(1+floor(fraction*T)):T);Y_valid=Y((1+floor(fraction*T)):T,1);
    mdl = fitglm(X_train(:,:)',Y_train,'linear','link','logit','Distribution','binomial');
    score_log = mdl.predict(X_valid(:,:)');
    [Xlog,Ylog,Tlog,AUClog(n_ens)] = perfcurve(Y_valid,score_log,'true');
end

