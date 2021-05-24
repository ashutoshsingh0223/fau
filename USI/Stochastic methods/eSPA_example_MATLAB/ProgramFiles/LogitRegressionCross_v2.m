function [out] = LogitRegressionCross_v2(X_train,Y_train,X_valid,Y_valid)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
   [B,FitInfo] = lassoglm(X_train',Y_train(1,:)','binomial','CV',2);
idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
B0 = FitInfo.Intercept(idxLambdaMinDeviance);
coef = [B0; B(:,idxLambdaMinDeviance)];
yhat = glmval(coef,X_valid','logit');
score_log = yhat;%(yhat>=0.5);
[Xlog,Ylog,Tlog,out.L_pred_valid] = perfcurve(Y_valid(1,:),double(score_log)',1);
out.coef=coef;
end

