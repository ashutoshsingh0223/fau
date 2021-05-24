function [out] = LDA(X_train,Y_train,X_valid,Y_valid)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
tic;
MdlLinear = fitcdiscr(X_train',Y_train(1,:)','discrimType','pseudoLinear');
score_log = predict(MdlLinear,X_valid');
[Xlog,Ylog,Tlog,out.L_pred_valid] = perfcurve(Y_valid(1,:),double(score_log)',1);
out.time=toc;
end
