function [out] = LogitRegressionCross(X_train,Y_train,X_valid,Y_valid)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    opts = statset('glmfit');
    opts.MaxIter = 1000;
    mdl = fitglm(X_train(:,:)',Y_train(1,:)','linear','link','logit','Distribution','binomial', 'options', opts);
    score_log = mdl.predict(X_valid(:,:)');
    [Xlog,Ylog,Tlog,out.L_pred_valid] = perfcurve(Y_valid(1,:),score_log,1);
end

