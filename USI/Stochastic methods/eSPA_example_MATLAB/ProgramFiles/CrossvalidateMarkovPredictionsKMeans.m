function [ err,P ] = CrossvalidateMarkovPredictionsKMeans(Pi,X,C,N_steps,fraction)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

Ng=numel(Pi);
for n=1:Ng
    [d,T]=size(X{n});
    for t=1:T-N_steps
        x(:,1,t)=X{n}(:,t);
        x(:,2,t)=X{n}(:,t+1);
        x(:,3,t)=X{n}(:,t+N_steps);
        pi(:,1,t)=Pi{n}(:,t);
        pi(:,2,t)=Pi{n}(:,t+1);
        pi(:,3,t)=Pi{n}(:,t+N_steps);
    end
end
T=size(x,3);
ind_perm=randperm(T);
x=x(:,:,ind_perm);
pi=pi(:,:,ind_perm);
T_train=round(T*fraction);
x_train=x(:,:,1:T_train);
pi_train=pi(:,:,1:T_train);
x_valid=x(:,:,(1+T_train):T);
pi_valid=pi(:,:,(1+T_train):T);

[ NT,P,fff ] = InferFullMarkovProbabilityJensen_Crossval(pi_train);
%[P] = lambdasolver_quadprog( pi_train );
%[ P,fff ] = InferFullMarkovExact_Crossval(pi_train);
err=EstimateMarkovPredictionError(P,C,pi_valid,x_valid,N_steps);
%err=EstimateMarkovPredictionError(P,C,pi_train,x_train,N_steps);

end

function [ P,fff ] = InferFullMarkovExact_Crossval(pi)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

tol=1e-10;
MaxIter=1500;
options=optimset('UseParallel','always',...
    'GradObj','on','Algorithm','sqp','MaxIter',MaxIter,'Display','iter','TolFun',tol,'TolCon',tol,'TolX',tol);
K=size(pi,1);
Aeq=zeros(K,K^2);beq=ones(K,1);
A=-eye(K^2);b=zeros(K^2,1);
for k=1:K
    Aeq(k,(K*(k-1)+1):k*K)=1;
end
[N_init,xxx_init] = InferFullMarkovProbabilityJensen_Crossval(pi);
%xxx_init=rand(K);
%for k=1:K
%   xxx_init(:,k)= xxx_init(:,k)./sum(xxx_init(:,k));
%end
xxx_init=reshape(xxx_init,1,K^2)';
[y]=FunEval_InferFullMarkovLogLikExact_Crossval(xxx_init,pi,K)

[xxx0,fff,flag,output] =  fmincon(@(x)FunEval_InferFullMarkovLogLikExact_Crossval...
    (x,pi,K)...
    ,xxx_init,(A),(b),Aeq,beq,[],[],[],options);
P=reshape(xxx0',K,K);
end

function [ NT,P,fff ] = InferFullMarkovProbabilityJensen_Crossval(pi)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

tol=1e-10;
MaxIter=1500;
%options=optimset('UseParallel','always',...
%    'GradObj','on','Algorithm','sqp','MaxIter',MaxIter,'Display','off','TolFun',tol,'TolCon',tol,'TolX',tol);
[K,d,T]=size(pi);
NT=zeros(K);
for t=1:T
    NT=NT+pi(:,1,t)*pi(:,2,t)';
end
P=zeros(K,K);
for j=1:K
    P(:,j)=NT(:,j)/sum(NT(:,j));
end
fff=sum(sum(log(NT.*P)));
end

function [y,dy]=FunEval_InferFullMarkovLogLikExact_Crossval(x,pi,K)
K2=K^2;
y=0;dy=zeros(1,K2);
ind_dy=1:K2;
TTT=size(pi,3);
for t=1:TTT
    pt=pi(:,1,t)*pi(:,2,t)';
    pt=reshape(pt,1,K2);
    %ind_dy=find((pt.*x')>1e-12);
    sd=sum(pt(ind_dy).*x(ind_dy)');
    y=y+log(sd);
    dy(ind_dy)=dy(ind_dy)+1./sd.*pt(ind_dy);
end
y=-y./TTT;
dy=-dy./TTT;
end

function err=EstimateMarkovPredictionError(P,C,pi,x,N_steps)
TTT=size(pi,3);
d=size(C,1);
P=C*P^N_steps;
%a=squeeze(x(:,3,:))-P*squeeze(pi(:,1,:));
%err=trace(a'*a);
err=0;
 for t=1:TTT
     err_vect=x(:,3,t)-P*pi(:,1,t);
     err=err+err_vect'*err_vect;
 end
err=err./(TTT*d);
end