function [ err] = CrossvalidateANNPredictionsParallel(in)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
pi=in.pi;
x=in.x;
C=in.C;
N_steps=in.N_steps;
fraction=in.fraction;
Stat_Size=in.Stat_Size;
ind_perm=in.perm;
flag=in.flag;

T=sum(Stat_Size);
x=x(:,:,ind_perm);
pi=pi(:,:,ind_perm);
T_train=round(T*fraction);
%x_train=x(:,:,1:T_train);
pi_train=pi(:,:,1:T_train);
x_valid=x(:,:,(1+T_train):T);
pi_valid=pi(:,:,(1+T_train):T);

net = patternnet(10);
net.trainParam.showWindow = 0;
net = train(net,squeeze(pi_train(:,1,:)),squeeze(pi_train(:,3,:)),'useParallel','no');
%view(net)
yy = net(squeeze(pi_valid(:,1,:)));
TTT=size(pi_valid,3);
err=0;
 for t=1:TTT
     err_vect=x_valid(:,3,t)-C*yy(:,t);
     err=err+err_vect'*err_vect;
 end
err=err./(TTT*size(C,1));


end

