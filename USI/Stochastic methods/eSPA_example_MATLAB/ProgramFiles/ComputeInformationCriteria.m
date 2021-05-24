function out=ComputeInformationCriteria(in,X)

[N_K,N_ens]=size(in);
[dim,T]=size(X);
N_eps=size(in{1,1}.C_full,3);
AICc=zeros(N_K,N_ens,N_eps);
phi_C=zeros(N_K,N_ens,N_eps);
N_param=zeros(N_K,N_ens,N_eps);
tol=1e-5;
for k=1:N_K
    KKK=size(in{k,1}.C,2);
    P_mean{k}=zeros(size(in{k,1}.P));
    C_mean{k}=zeros(size(in{k,1}.C));
    C_full{k}=zeros(dim,KKK,size(in{k,1}.C_full,3));
    NN_mean{k}=zeros(size(in{k,1}.P));

    for n=1:N_ens
        [~,ind_sort]=AllignClusterCenters(in{k,1}.C,in{k,n}.C);
        P_mean{k}=P_mean{k}+(1/N_ens)*in{k,n}.P(:,ind_sort);
        input_NN=eye(size(in{k,n}.C,2));
        NN_mean{k}=NN_mean{k}+(1/N_ens)*in{k,n}.net(input_NN(:,ind_sort));
        C_mean{k}=C_mean{k}+(1/N_ens)*in{k,n}.C(:,ind_sort);
        %L_discr_valid(k,n)=in{k,n}.L_discr_pred;

        for n_eps=1:size(in{k,n}.C_full,3)                    
            N_param(k,n,n_eps)=KKK*dim;
            CC=AllignClusterCenters(in{k,1}.C_full(:,:,1),in{k,n}.C_full(:,:,n_eps));
            C_full{k}(:,:,n_eps)=C_full{k}(:,:,n_eps)+(1/N_ens)*CC;
            for d=1:dim
                for k1=1:KKK
                    flag=0;
                    for k2=k1+1:KKK
                        C_dist=(CC(d,k1)-CC(d,k2))^2;
                        phi_C(k,n,n_eps)=phi_C(k,n,n_eps)+C_dist;
                        if C_dist<tol
                           flag=1;
                        end
                    end
                    if flag==1
                    N_param(k,n,n_eps)=N_param(k,n,n_eps)-1;
                    end
                end
            end
            AICc(k,n,n_eps)=2*in{k,n}.L_full(n_eps)*T*dim+2*N_param(k,n,n_eps)...
                + 2*N_param(k,n,n_eps)*(N_param(k,n,n_eps)+1)/(T-N_param(k,n,n_eps)-1);
            BIC(k,n,n_eps)=2*in{k,n}.L_full(n_eps)*T*dim+N_param(k,n,n_eps)*log(T);
        end
    end
end
AICc_final=zeros(N_K,N_eps);
BIC_final=zeros(N_K,N_eps);
N_param_final=zeros(N_K,N_eps);
phi_C_final=zeros(N_K,N_eps);
%L_discr_valid_final=zeros(N_K,N_eps);
%L_pred_valid_final=zeros(N_K,N_eps);
N=1/N_ens;
for n=1:N_ens
    AICc_final=AICc_final+N*squeeze(AICc(:,n,:));
    BIC_final=BIC_final+N*squeeze(BIC(:,n,:));
    N_param_final=N_param_final+N*squeeze(N_param(:,n,:));
    phi_C_final=phi_C_final+N*squeeze(phi_C(:,n,:));
    %L_discr_valid_final=L_discr_valid_final+N*squeeze(phi_C(:,n,:));
end
out.AICc_final=AICc_final;
out.BICc_final=BIC_final;
out.phi_C_final=phi_C_final;
out.N_param_final=N_param_final;
out.P_mean=P_mean;
out.NN_mean=NN_mean;
out.C_mean=C_mean;
out.C_full_mean=C_full;