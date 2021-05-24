function [ mean_err,mean_P] = ParallelMarkovEstimationAndPrediction(Pi, C_pred,N_steps,Y,fraction,flag,N_anneal)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
Ng=numel(Y);
Stat_Size=zeros(1,Ng);
for n=1:Ng
    Stat_Size(n)=size(Y{n},2)-N_steps;
end
kk=1;x=zeros(size(Y{1},1),3,sum(Stat_Size));
pi=zeros(size(Pi{1},1),3,sum(Stat_Size));
for n=1:Ng
    [d,T]=size(Y{n});
    for t=1:T-N_steps
        x(:,1,kk)=Y{n}(:,t);
        x(:,2,kk)=Y{n}(:,t+1);
        x(:,3,kk)=Y{n}(:,t+N_steps);
        pi(:,1,kk)=Pi{n}(:,t);
        pi(:,2,kk)=Pi{n}(:,t+1);
        pi(:,3,kk)=Pi{n}(:,t+N_steps);
        kk=kk+1;
    end
end

for n_ann=1:N_anneal
    %n_ann
    in{n_ann}.C=C_pred;
    in{n_ann}.N_steps=N_steps;
    in{n_ann}.x=x;
    in{n_ann}.pi=pi;
    in{n_ann}.flag=flag;
    in{n_ann}.Stat_Size=Stat_Size;
    in{n_ann}.fraction=fraction;
    in{n_ann}.perm=randperm(sum(Stat_Size));
end
parfor n_ann=1:N_anneal
    [ err{n_ann},P_full{n_ann}] = CrossvalidateMarkovPredictionsParallel(in{n_ann});
end
mean_err=0;mean_P=0;
for n_ann=1:N_anneal
    mean_err=mean_err+err{n_ann};
    mean_P=mean_P+P_full{n_ann};
end
mean_P=mean_P./N_anneal;
mean_err=mean_err./N_anneal;
end

