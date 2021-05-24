function []= mouse_lick_(X,pi,interval,out)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
d=size(out.C,1);
K=size(out.C,2);



N_interval=length(interval);
p=zeros(1,N_interval-1);pp=p;pm=p;p0=p;
for i=1:length(interval)-1
    clear gamma_valid pi_valid 
ii1=find(and(and(X(1,:)==0,and(interval(i)<X(3,:),X(3,:)<interval(i+1))),pi(1,:)==1));
ii0=find(and(and(X(1,:)==0,and(interval(i)<X(3,:),X(3,:)<interval(i+1))),pi(1,:)==0));
p(i)=length(ii1)/(length(ii0)+length(ii1));
[p0(i),pp(i),pm(i)] = WilsonScore(p(i),length(ii0)+length(ii1));
%if length(ii1)>1
X_valid=X(:,[ii0 ii1]);%[0;mean(X(2,[ii0 ii1]));0.5*(interval(i+1)+interval(i));mean(X(4:63,[ii0 ii1])')'];
%else
% X_valid=[0;mean(X(2,ii1));0.5*(interval(i+1)+interval(i));X(4:63,ii1)];   
%end
[gamma_valid,LLL2]=EvaluateGammaParallel_Lukas_vect(diag(sqrt(out.W))*X_valid,out.C,1,d,K,[],0);
pi_valid=out.P*gamma_valid;
p_eSPA(i)=mean(pi_valid(1,:));
end
x_interval=0.5*(interval(1:N_interval-1)+interval(2:N_interval));
figure;errorbar(x_interval,p0,p0-pm,pp-p0,'--o','LineWidth',3)
hold on;
plot(x_interval,p_eSPA,'-','LineWidth',3)
end

