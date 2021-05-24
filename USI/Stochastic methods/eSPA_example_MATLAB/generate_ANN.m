clc

clear all
close all

N_neurons=1000;
T_out=150;

T=10000;
P0=0.05;
P_low=0.0;
P_high=0.9;
N_imp=2;




mean_time=30;
margin_low=10;
margin_high=10;
sigma(:,:,1)=5;sigma(:,:,2)=sigma(:,:,1);sigma(:,:,3)=4*sigma(:,:,1);sigma(:,:,4)=4*sigma(:,:,1);
gm = gmdistribution([10 25 50 200]',sigma,[5/10 3/10 1.5/10 0.5/10]);
figure;plot([0:0.01:250],pdf(gm,[0:0.01:250]'),'LineWidth',3)
set(gca,'FontSize',20,'LineWidth',2);

clock=zeros(N_neurons,1);
for t=1:T
    for d=1:N_neurons
        clock(d)=clock(d)+ceil(max(1,random(gm,1)));
        r(d,clock(d))=1;
    end
end
r=r(:,1:min(clock));
td=TransformTS2TD(r);
figure;histogram(td(1,:))

T=size(r,2);
pi=zeros(1,T);
for t=1:T
         pd=makedist('normal','mu',mean_time,'sigma',4);
         p=pdf(pd,sum(td(1:N_imp,t)))/pdf(pd,mean_time);
         pd=makedist('Binomial','p',p);
         pi(t)=random(pd,1);
end
% pi=zeros(1,T);
% for t=1:T
%     if and(sum(td(1:N_imp,t))>mean_time-margin_low,sum(td(1:N_imp,t))<mean_time+margin_high)
%         %if sum((td(1:N_imp,t)-mean_time).^2)<N_imp*margin_low.^2
%         pd=makedist('Binomial','p',P_high);
%         pi(t)=random(pd,1);
%     else
%         pd=makedist('Binomial','p',P_low);
%         pi(t)=random(pd,1);
%     end
% end

% pi=zeros(1,T);
% for t=1:T
%     for d=1:N_imp
%        flag=1;
%        s=t;
%        while and(flag,s>0)
%            if r(d,s)==1
%               time_delay(d)=t-s;
%               flag=0;
%            end                  
%            s=s-1;
%        end
%        if s==0
%           time_delay(d)=mean_time; 
%        end
%     end 
%     if and(sum(time_delay)>mean_time*N_imp-margin,sum(time_delay)<mean_time*N_imp+margin)
%         pd=makedist('Binomial','p',P_high);
%         pi(t)=random(pd,1);
%     else
%         pd=makedist('Binomial','p',P_low);
%         pi(t)=random(pd,1);
%     end
% end
figure;plot(pi(1:1000),':.')
sum(pi)

[~,ind1]=find(pi==1);ind=randperm(length(ind1));ind1=ind1(ind(1:T_out));
[~,ind2]=find(pi==0);ind=randperm(length(ind2));ind2=ind2(ind(1:T_out));
X=[(td(:,ind1)) (td(:,ind2))];
clear pi
pi(1,:)=[ones(1,length(ind1)) zeros(1,length(ind2))];
pi(2,:)=1-pi(1,:);
figure;imagesc(X)
figure;plot3(X(1,1:T_out),X(2,1:T_out),X(3,1:T_out),'ro','MarkerSize',9);
hold on;plot3(X(1,1+T_out:2*T_out),X(2,1+T_out:2*T_out),X(3,1+T_out:2*T_out),'bx','MarkerSize',9);

figure;subplot(1,3,1);plot(X(1,1:T_out),X(2,1:T_out),'ro','MarkerSize',9);
hold on;plot(X(1,1+T_out:2*T_out),X(2,1+T_out:2*T_out),'bx','MarkerSize',9);
xlabel('Spike Delays for Neuron 1');ylabel('Spike Delays for Neuron 2');
set(gca,'FontSize',20,'LineWidth',2);
subplot(1,3,2);plot(X(1,1:T_out),X(3,1:T_out),'ro','MarkerSize',9);
hold on;plot(X(1,1+T_out:2*T_out),X(3,1+T_out:2*T_out),'bx','MarkerSize',9);
xlabel('Spike Delays for Neuron 1');ylabel('Spike Delays for Neuron 3');
set(gca,'FontSize',20,'LineWidth',2);
subplot(1,3,3);plot(X(2,1:T_out),X(3,1:T_out),'ro','MarkerSize',9);
hold on;plot(X(2,1+T_out:2*T_out),X(3,1+T_out:2*T_out),'bx','MarkerSize',9);
xlabel('Spike Delays for Neuron 2');ylabel('Spike Delays for Neuron 3');
set(gca,'FontSize',20,'LineWidth',2);
save Input/NeuralModelExample_1.mat X pi

function td=time_delays(r,ind);

for i=1:length(ind)
    ind=[];
    for t=1:size(r,2)
        if r(i,t)==1
            ind=[ind t];
        end
    end
    td{i}=diff(ind);
end
end

function td=TransformTS2TD(r)
[D,T]=size(r);
td=zeros(D,T);
clock=zeros(D,1);
for t=1:T
    for d=1:D
        if r(d,t)==1
            clock(d)=0;
        else
            clock(d)=clock(d)+1;
        end
    end    
    td(:,t)=clock;
end
end