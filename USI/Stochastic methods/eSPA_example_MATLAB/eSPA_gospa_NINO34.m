%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Analysis of the NINO3.4 predictions, SSA data are 100 PCs from NOA SSA, 
%% delta_Z data prepared by Terry O'Kane (CSIRO).
%% Analysed data is in the file 'enso_predict.mat'
%% Variable X contain 6 201xT values of 101 dominant SSA PCs (first 101 rows of X) over T months and 
%% 100 dominant PCs of delta_Z  anomailies over T months (the rows 102 to 201 of X)
%% Variable pi contain 2XT probabilty matrix of NINO3.4>0.4 at t+24months (pi(2,t)=1 if NIN03.4(t+24)>0 and pi(1,t)=0 otherwise)
%% 
%% (c) Illia Horenko 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clc
clear all
close all
rand('seed',1);
randn('seed',1);

addpath('ProgramFiles/')
addpath('Output/')
addpath('Input/')
%% Set this flag to 1 if you have the licence for a "Parallel Computing" toolbox of MATLAB
flag_parallel=1;
%% Set this flag_DL to 1 if you would like to train the Deep Learning classifier and have a
%% "Deep Learning" toolbox licence of MATLAB
flag_DL=1;
%% Set this flag_goSPA to 1 if you would like to train the gauge-optimized SPA classifier 
flag_goSPA=0;
%% Load the feature matrix X and matrix of label probabilities pi
load('enso_predict_24months.mat')
%% Number of eSPA patterns/boxes/clusters
K=50;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T=size(X,2);
%% Number of annealing steps to avoid trapping in the local optimum
%% (take it as Number_of_Parallel_Cores*N, where N is any positive integer)
N_anneal=48;
%% Select flag_AUC=1 to use Area Under Curve as a performance metrics
%% selecting flag_AUC=0 implies an Accuracy as a performance metrics
flag_AUC=0;
%% 
fraction=3/4;
%% Normalize the data and make it dimensionless
ind=[];
%% T_detrend_start indicates start time for linear detrending
%% T_detrend_start is set to 300 since looking at the rows of the matrix X
%% it seems that the linear trend in PCs starts after around 300 months after
%% start of the analyzed time series, corresponding to the year 1975 
T_detrend_start=300;
for i=1:size(X,1)
    X(i,:)=X(i,:)-mean(X(i,:));
    if max(abs(X(i,:)))==0;
        ind=[ind i];
    else
        X(i,:)=X(i,:)/max(abs(X(i,:)));
    end
    X(i,1:T_detrend_start)=detrend(X(i,1:T_detrend_start),0);
    X(i,(T_detrend_start+1):T)=detrend(X(i,(T_detrend_start+1):T),1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ind_perm=1:T;%randperm(T);
[~,ind_back]=sort(ind_perm,'ascend');
X=X(:,ind_perm);
pi=pi(:,ind_perm);
X_train=X(:,1:floor(fraction*T));pi_train=pi(:,1:floor(fraction*T));
X_valid=X(:,(1+floor(fraction*T)):T);pi_valid=pi(:,(1+floor(fraction*T)):T);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set a grid of eSPA parameters
%% Parameter 'reg_param_W' sets the weight of the entropic feature selection subproblem (decreasing this value
%% will reduce the number of entropically-selected features)
%% Parameter 'reg_param_CL' sets the relative importance of classification subproblem compared
%% to the discretization and entropic feature selection subproblems
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
reg_param_W=[6e-3 8e-3 1e-2 1.2e-2 ];
reg_param_CL=[1 1e-6  1e-5  1e-4 2e-4 3e-4 4e-4 6e-4 1e-3];
k=1;
for i=1:length(reg_param_W)
    for j=1:length(reg_param_CL)
        reg_param(:,k)=[reg_param_W(i);reg_param_CL(j)];
        k=k+1;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set a grid of goSPA parameters
%% Parameter 'gauge_param_Dim' sets the reduced dimensionality of the orthogonal gauge rotation
%% Parameter 'gauge_param_CL' sets the relative importance of classification subproblem compared
%% to the discretization and gauge rotation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    gauge_param_Dim=[70];
    gauge_param_CL=[1e-5 5e-4 1e-3 2e-3 5e-3];
    k=1;
    for i=1:length(gauge_param_Dim)
        for j=1:length(gauge_param_CL)
            gauge_param(:,k)=[gauge_param_Dim(i);gauge_param_CL(j)];
            k=k+1;
        end
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




seed=rng;
out_eSPA = SPACL_kmeans_dim_entropy_analytic_v2(X_train,pi_train,...
    K,N_anneal,[],reg_param,X_valid,pi_valid,1,flag_AUC,flag_parallel)
if flag_AUC==1
disp(['eSPA: AUC on validation data ' num2str(-out_eSPA.L_pred_valid_Markov) ', computational time ' num2str(out_eSPA.time_Markov) ' seconds, has ' num2str(max(out_eSPA.N_params_Markov)) ' tuneable parameters']);
else
disp(['eSPA: Accuracy on validation data ' num2str(-out_eSPA.L_pred_valid_Markov) ', computational time ' num2str(out_eSPA.time_Markov) ' seconds, has ' num2str(max(out_eSPA.N_params_Markov)) ' tuneable parameters']);
end   
gamma=[out_eSPA.gamma out_eSPA.gamma_valid];
pi_pred=out_eSPA.P*out_eSPA.gamma_valid;
hf=figure;plot(pi_valid(2,:),'k-','LineWidth',2);hold on;plot(pi_pred(2,:),'r--o','LineWidth',3);ylim([-0.05 1.05])
legend('NINO3.4>0.4','eSPA 24-months ahead prediction');
xlabel('Time, months');
ylabel('Probability on Validation Data');
set(gca,'FontSize',20,'LineWidth',2);

populations=sum(gamma,2)/size(gamma,2);
[~,ii]=sort(populations,'descend');kk=find(populations(ii)>0);
ii=ii(kk);

figure;plot(out_eSPA.W,'.:','LineWidth',2,'MarkerSize',9);
xlabel('Feature Index, d');
ylabel('eSPA Feature Weight, W(d)')
set(gca,'FontSize',20,'LineWidth',2);

figure;imagesc(out_eSPA.gamma_valid(ii,:));caxis([0 1]);title('Probablities P(i,t) for validation data to belong to eSPA patterns')
xlabel('Validation Data Index, t');
ylabel('eSPA Pattern Index, i')
set(gca,'FontSize',20,'LineWidth',2);
set(gcf,'Position',[10 100 800  600])

figure;subplot(3,1,1); 
plot(populations(ii),'.:','LineWidth',2,'MarkerSize',9)
caxis([0 1]);ylabel({'Proportion of data', 'in this eSPA pattern'})
xlabel('eSPA Pattern Index, i');
set(gca,'FontSize',20,'LineWidth',2);
subplot(3,1,2); 
plot(out_eSPA.P_valid(1,ii),'.:','LineWidth',2,'MarkerSize',9)
caxis([0 1]);ylabel('Probablity of Label=1')
xlabel('eSPA Pattern Index, i');ylim([-0.05 1.05])
set(gca,'FontSize',20,'LineWidth',2);
subplot(3,1,3);
clear C
for d=1:length(out_eSPA.W);
    C(d,:)=out_eSPA.W(d)*out_eSPA.C(d,:);
end
imagesc(C(:,(ii)))
ylabel('Feature Index,d')
xlabel('eSPA Pattern Index, i');
zlabel('Value')
set(gca,'FontSize',20,'LineWidth',2);
set(gcf,'Position',[10 100 900  800])

seed=rng;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if flag_goSPA==1
    %K=20;
    out_goSPA = SPACL_kmeans_dim_entropy_analytic_gauge(X_train,pi_train,...
        K,5*N_anneal,[],gauge_param,X_valid,pi_valid,1,flag_AUC,flag_parallel,seed)
    if flag_AUC==1
        disp(['goSPA: AUC on validation data ' num2str(-out_goSPA.L_pred_valid_Markov) ', computational time ' num2str(out_goSPA.time_Markov) ' seconds, has ' num2str(max(out_goSPA.N_params_Markov)) ' tuneable parameters']);
    else
        disp(['goSPA: Accuracy on validation data ' num2str(-out_goSPA.L_pred_valid_Markov) ', computational time ' num2str(out_goSPA.time_Markov) ' seconds, has ' num2str(max(out_goSPA.N_params_Markov)) ' tuneable parameters']);
    end
    pi_pred=out_goSPA.P*out_goSPA.gamma_valid;
    figure(hf);hold on;plot(pi_pred(2,:),'g:h');ylim([-0.05 1.05])
else
        out_goSPA=[];
end


if flag_DL==1
    %% From here you will need the "Deep Learning" Toolbox of MATLAB
    
    out_DL= DeepNN_Classify(X_train,pi_train, X_valid, pi_valid,flag_AUC);
    [~,i]=max(-out_DL.L_pred_valid);
    if flag_AUC==1
        disp(['Deep Learning (LSTM): AUC on validation data ' num2str(max(-out_DL.L_pred_valid(i))) ', computational time ' num2str(max(out_DL.time(i))) ' seconds, has ' num2str(max(out_DL.N_params(i))) ' tuneable parameters']);
    else
        disp(['Deep Learning (LSTM): Accuracy on validation data ' num2str(max(-out_DL.L_pred_valid(i))) ', computational time ' num2str(max(out_DL.time(i))) ' seconds, has ' num2str(max(out_DL.N_params(i))) ' tuneable parameters']);
    end
    disp(['Deep Learning (LSTM) is here ' num2str(max(out_DL.time(length(out_DL.time)))/out_eSPA.time_Markov)...
        ' times more costly then eSPA']);
    
    out_DL.net{i}.Layers
    clear pi_pred
    for t=1:size(X_valid,2)
        pi_pred(:,t)=predict(out_DL.net{i},X_train(:,t))';
    end
    figure(hf);hold on;plot(pi_pred(2,:),'m:.','LineWidth',1);ylim([-0.05 1.05])
   legend('NINO3.4>0.4','eSPA 24-months ahead prediction','Deep Learning 24-months ahead prediction');
else
    out_DL=[];
end
save Output/NINO34_out.mat out_DL out_eSPA out_goSPA

