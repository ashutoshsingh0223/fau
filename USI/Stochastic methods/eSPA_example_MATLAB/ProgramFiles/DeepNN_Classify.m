function [out_kmeans]= DeepNN_Classify(X,pi, X_valid, pi_valid,flag_AUC)

[d,T]=size(X);T_valid=size(X_valid,2);
T_interm=floor(9/10*T);
for t=1:T_interm
    XTrain{t,1}=X(:,t);
    [~,YTrain(t,1)]=max(pi(:,t));
end
k=1;
for t=(1+T_interm):T
    XTrain_i{k,1}=X(:,t);
    [~,YTrain_i(k,1)]=max(pi(:,t));
    k=k+1;
end
for t=1:T_valid
    XValidation{t,1}=X_valid(:,t);
    [~,YValidation(t,1)]=max(pi_valid(:,t));
end
YTrain=categorical(YTrain);
YTrain_i=categorical(YTrain_i);
YValidation=categorical(YValidation);

numFeatures = d;
numClasses = size(pi,1);
N=[2 4 10 50 100 200];
for n=1:length(N)
 %   n
numHiddenUnits = N(n);

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
miniBatchSize = 20;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',1000, ...
    'MiniBatchSize',miniBatchSize, ...
    'ValidationData',{XTrain_i,YTrain_i}, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','none');

tic;net = trainNetwork(XTrain,YTrain,layers,options);
%[scores] = predict(net,XValidation);
gamma_valid=double(net.predict(XValidation))';
%gamma_valid=zeros(2,length(scores));
%gamma_valid(1,find(scores==1))=1;gamma_valid(2,find(scores==2))=1;
out_kmeans.time(n)=toc;
[L_pred_test] = AUC_of_Prediction(gamma_valid,eye(size(pi_valid,1)),pi_valid,0,flag_AUC);
out_kmeans.L_pred_valid(n)=L_pred_test/(size(pi_valid,2)*size(pi_valid,1));
out_kmeans.net{n}=net;
out_kmeans.N_params(n)=prod(size(net.Layers(2).InputWeights))+...
    prod(size(net.Layers(2).RecurrentWeights))+prod(size(net.Layers(2).Bias))+prod(size(net.Layers(3).Weights))...
    +prod(size(net.Layers(3).Bias));
end



