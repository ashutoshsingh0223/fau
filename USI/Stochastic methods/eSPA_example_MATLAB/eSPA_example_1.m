%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Synthetic data example 1 from the paper
%% "On a Scalable Entropic Breaching of the Overfitting Barrier for Small Data Problems in Machine Learning"
%% Illia Horenko
%% Neural Computation August 2020, Vol. 32, No. 8, pp. 1563–1579
%%
%% This synthetic example is motivated by the Seq-Data application model from longevity research
%% (c) Illia Horenko 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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
%% Set this flag to 1 if you would like to train the Deep Learning classifier and have a
%% "Deep Learning" toolbox licence of MATLAB
flag_DL=1;
%% Load the feature matrix X and matrix of label probabilities pi
%load('classification_SNP_buche.mat')
%load('classification_WDBC.mat')
%% Number of eSPA patterns/boxes/clusters
K=3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Number of annealing steps to avoid trapping in the local optimum
%% (take it as Number_of_Parallel_Cores*N, where N is any positive integer)
N_anneal=16;
%% Select flag_AUC=1 to use Area Under Curve as a performance metrics
%% selecting flag_AUC=0 implies an Accuracy as a performance metrics
flag_AUC=1;
%% 
fraction=4/5;

%% Set the statistics size and dimension for the synthetic data example
time_statistics=50;
dimension=51;
for dim_index=1:length(dimension)
        T=time_statistics;sigma=5;d=dimension(dim_index);
        [X,pi]=generate_challenge1(T,sigma,d);
         disp(['statistics length: ' num2str(size(X,2)) '; dimension: ' num2str(d)]);

%% Normalize the data and make it dimensionless
        ind=[];
        for i=1:size(X,1)
            X(i,:)=X(i,:)-mean(X(i,:));
            if max(abs(X(i,:)))==0;
                ind=[ind i];
            else
                X(i,:)=X(i,:)/max(abs(X(i,:)));
            end
        end
        XX{dim_index}=X(setdiff(1:size(X,1),ind),:);
        PI{dim_index}=pi;
end

%% Set a grid of eSPA model parameters

reg_param_W=[1e-3 4e-3 8e-3 16e-3 32e-3 1000];
reg_param_CL=[1e-5 2e-5 4e-5 1e-4 5e-4 1e-3 5e-3 1e-2];
k=1;
for i=1:length(reg_param_W)
    for j=1:length(reg_param_CL)
        reg_param(:,k)=[reg_param_W(i);reg_param_CL(j)];
        k=k+1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T=size(X,2);
ind=1;

figure;subplot(1,2,1);[~,i]=find(PI{ind}(1,:)==1);plot(XX{ind}(1,i),XX{ind}(2,i),'r+','LineWidth',2);hold on;
subplot(1,2,2);plot(XX{ind}(3,i),XX{ind}(4,i),'r+','LineWidth',2);hold on;
subplot(1,2,1);[~,i]=find(PI{ind}(2,:)==1);plot(XX{ind}(1,i),XX{ind}(2,i),'bx','LineWidth',2);legend('normal','long-living');
xlabel('Expression of Gene 1 (mTOR)');ylabel('Expression of Gene 2 (IIS)');
set(gca,'FontSize',20,'LineWidth',2);
subplot(1,2,2);plot(XX{ind}(3,i),XX{ind}(4,i),'bx','LineWidth',2);hold on;legend('normal','long-living');
set(gca,'FontSize',20,'LineWidth',2);
xlabel('Expression of Gene 3');ylabel('Expression of Gene 4');
set(gcf,'Position',[10 100 1000  500])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ind_perm=randperm(T);
[~,ind_back]=sort(ind_perm,'ascend');
clear X pi
X=XX{ind}(:,ind_perm);
pi=PI{ind}(:,ind_perm);
X_train=X(:,1:floor(fraction*T));pi_train=pi(:,1:floor(fraction*T));
X_valid=X(:,(1+floor(fraction*T)):T);pi_valid=pi(:,(1+floor(fraction*T)):T);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



seed=rng;
out_eSPA = SPACL_kmeans_dim_entropy_analytic_v2(X_train,pi_train,...
    K,N_anneal,[],reg_param,X_valid,pi_valid,1,flag_AUC,flag_parallel)
if flag_AUC==1
disp(['eSPA: AUC on validation data ' num2str(-out_eSPA.L_pred_valid_Markov) ', computational time ' num2str(out_eSPA.time_Markov) ' seconds, has ' num2str(max(out_eSPA.N_params_Markov)) ' tuneable parameters']);
else
disp(['eSPA: Accuracy on validation data ' num2str(-out_eSPA.L_pred_valid_Markov) ', computational time ' num2str(out_eSPA.time_Markov) ' seconds, has ' num2str(max(out_eSPA.N_params_Markov)) ' tuneable parameters']);
end    
figure;plot(out_eSPA.W,'.:','LineWidth',2,'MarkerSize',9);
xlabel('Feature Index, d');
ylabel('eSPA Feature Weight, W(d)')
set(gca,'FontSize',20,'LineWidth',2);

figure;imagesc(out_eSPA.gamma_valid);caxis([0 1]);title('Probablities P(i,t) for validation data to belong to eSPA patterns')
xlabel('Validation Data Index, t');
ylabel('eSPA Pattern Index, i')
set(gca,'FontSize',20,'LineWidth',2);
set(gcf,'Position',[10 100 800  600])

figure;subplot(2,1,1); 
plot(out_eSPA.P_valid(1,:),'.:','LineWidth',2,'MarkerSize',9)
caxis([0 1]);ylabel('Probablity of Label=1')
xlabel('eSPA Pattern Index, i');
set(gca,'FontSize',20,'LineWidth',2);xlim([0.5 size(out_eSPA.C,2)+0.5])
subplot(2,1,2);
clear C
for d=1:length(out_eSPA.W);
    C(d,:)=out_eSPA.W(d)*out_eSPA.C(d,:);
end
plot(C)
ylabel('Feature Index,d')
xlabel('eSPA Pattern Index, i');
zlabel('Value')
set(gca,'FontSize',20,'LineWidth',2);
set(gcf,'Position',[10 100 800  600])
legend

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
else
    out_DL=[];
end

save(['Output/example_out_dimension_' num2str(dimension+1) '.mat'], 'out_DL', 'out_eSPA')
