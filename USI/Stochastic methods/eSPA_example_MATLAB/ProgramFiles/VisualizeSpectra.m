function [] = VisualizeSpectra(out,k,N_ens,mm,S,U,mmu)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[D,K]=size(out{k,1}.C);
M=size(out{k,1}.P,1);
for ind_K=1:M
    labels{ind_K}=num2str(ind_K);
end

%[N_ens,N_K]=size(out)
if nargin>3
    U=U*S;
end
C1=out{k,1}.C;
pi_out=zeros(M,K);
C=(1/N_ens)*C1;

for ind_K=1:K
    pi=zeros(K,1);pi(ind_K)=1;
    pi_out(:,ind_K)=out{k,1}.net(pi);
end
pi_out=(1/N_ens)*pi_out;


for n=2:N_ens
    C2=out{k,n}.C;
    %C2=mmu+U*(mm.*out{k,n}.C);
    [C2,ind_perm]=AllignClusterCenters(C1,C2);
    C=C+(1/N_ens)*C2;
    for ind_K=1:K
        pi=zeros(K,1);pi(ind_K)=1;
        pi=pi(ind_perm);
        pi_out(:,ind_K)=pi_out(:,ind_K)+(1/N_ens)*out{k,n}.net(pi);
    end
end
if nargin>3
Cfin=zeros(size(mmu,1),K);
for kk=1:K
    Cfin(:,kk)=mmu(:,1)+U*(mm'.*C(:,kk));
end
else
   Cfin=C; 
end

figure;
for kk=1:K
    subplot(ceil(K/2),2,kk);plot(Cfin(:,kk));
end
figure;
for kk=1:K
    subplot(ceil(K/2),2,kk);pie(pi_out(:,kk),labels);
end
end

