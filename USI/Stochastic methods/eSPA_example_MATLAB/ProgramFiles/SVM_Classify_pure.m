function [out_kmeans]= SVM_Classify_pure(X,pi, X_valid, pi_valid,flag_AUC) 

[d,T]=size(X);m=size(pi,1);T_valid=size(X_valid,2);
tic;
time_pca=toc;
N_Params_pca=0;%prod(size(V))+prod(size(mu));

tic;
grp(1,find(pi(1,:)==1))=1;grp(1,find(pi(1,:)==0))=-1;
    opts = struct('Optimizer','bayesopt','ShowPlots',false,...
    'AcquisitionFunctionName','expected-improvement-plus');
svmmod = fitcsvm(X',grp,'KernelFunction','rbf',...
    'OptimizeHyperparameters','auto','Verbose',0,'IterationLimit',1e8,'NumPrint',10000,'HyperparameterOptimizationOptions',opts)
    out_kmeans.N_params=N_Params_pca+ prod(size(svmmod.Alpha))+prod(size(svmmod.Bias));   
    [xx] = predict(svmmod,X_valid');
    out_kmeans.time=time_pca+toc;
    out_kmeans.gamma_valid(1,:)=(xx==1)';out_kmeans.gamma_valid(2,:)=1-out_kmeans.gamma_valid(1,:);
    [L_pred_test] = AUC_of_Prediction(out_kmeans.gamma_valid,eye(2),pi_valid,0,flag_AUC);
    out_kmeans.L_pred_valid=L_pred_test/(size(pi_valid,2)*size(pi_valid,1));
    
    
 