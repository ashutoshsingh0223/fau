function [N,T,E,T_std,E_std,K_nonzero] = ExtractObservables4Plot(out,flag_bayes,N_max,index,K)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
if nargin<=2
    N_max=size(out,2);
end
CI=0.95;
K_nonzero=[];
N_tresh=1;
if strcmp(flag_bayes,'spa')
    NN=numel(out);
    %E=zeros(1,length(K));N=E;T=E;T_std=E;E_std=E;
    kkk=1;
    for k=1:length(K);
        for n=1:NN;
            [~,ii]=find(sum(out{n}{k}.gamma,2)>0);
            KK=length(ii);
            if or(length(intersect(KK,K_nonzero))==0,kkk==1)
                K_nonzero=[K_nonzero KK];
                ind_K(kkk)=length(K_nonzero);
            else
                [~,ind_K(kkk)]=find(KK==K_nonzero);
            end
            ind_k(kkk)=k;
            ind_n(kkk)=n;
            kkk=kkk+1;
        end;
    end;
    
    for kkk=1:length(K_nonzero)
        NNN(kkk)=sum(ind_K==kkk);
    end
    [~,nnn]=find(NNN>=N_tresh);
    E=0*NNN;T=0*NNN;
    for t=1:length(ind_K);
        E(ind_K(t))=E(ind_K(t))+out{ind_n(t)}{ind_k(t)}.L_pred_valid_Markov/NNN(ind_K(t));
        %Rep(n,k)=out{n}{k}.L_pred_valid_Markov;
        if size(out{ind_n(t)}{ind_k(t)}.W,2)==1
        N(ind_K(t))=K_nonzero(ind_K(t))*(size(out{ind_n(t)}{ind_k(t)}.P,1)-1)+K_nonzero(ind_K(t))*(size(out{ind_n(t)}{ind_k(t)}.C,1)+1);%out{n}{k}.N_params_Markov/NN;
        else
        N(ind_K(t))=K_nonzero(ind_K(t))*(size(out{ind_n(t)}{ind_k(t)}.P,1)-1)+K_nonzero(ind_K(t))*size(out{ind_n(t)}{ind_k(t)}.C,1)+prod(size(out{ind_n(t)}{ind_k(t)}.W));%out{n}{k}.N_params_Markov/NN;    
        end
        T(ind_K(t))=T(ind_K(t))+out{ind_n(t)}{ind_k(t)}.time_Markov/NNN(ind_K(t));
    end;
    E=E(nnn);N=N(nnn);T=T(nnn);K_nonzero=K_nonzero(nnn);
    [~,iii]=sort(N,'ascend');
    E=E(iii);N=N(iii);T=T(iii);K_nonzero=K_nonzero(iii);
    T_std=[];E_std=[];
    %[E_pl,E_mi]=EmpConfIntArray(E,Rep,CI);
    %[~,ii]=sort(N,'ascend');
    %N=N(ii);
    %E=E(ii);
    %T=T(ii);
else
    if nargin<4
        if and(numel(out{1})>1,flag_bayes=='bayes')
            index=1:length(out{1}{1}.N_params_Markov);
        elseif and(numel(out{1})>1,flag_bayes~='bayes')
            index=1:length(out{1}{1}.N_params);
        else
            index=1:length(out{1}.N_params);
        end
    end
    N_params=[];
    cpu_time=[];
    Pred_Error=[];
    if flag_bayes=='bayes'
        for i=1:N_max
            for j=1:size(out{1},2)
                if isfield(out{i}{j},'time')
                    if length(out{i}{j}.time)==1
                        N_params=[N_params out{i}{j}.N_params_Markov*ones(1,length(index))];
                        cpu_time=[cpu_time out{i}{j}.time_Markov*ones(1,length(index))];
                        Pred_Error=[Pred_Error out{i}{j}.L_pred_valid_Markov*ones(1,length(index))];
                    else
                        N_params=[N_params out{i}{j}.N_params_Markov(index)];
                        cpu_time=[cpu_time out{i}{j}.time_Markov(index)];
                        Pred_Error=[Pred_Error out{i}{j}.L_pred_valid_Markov(index)];
                    end
                end
            end
        end
    else
        for i=1:N_max
            if size(out{1},2)>1
                for j=1:size(out{1},2)
                    if isfield(out{i}{j},'time')
                        N_params=[N_params out{i}{j}.N_params(index)];
                        if length(out{i}{j}.time)==1
                            cpu_time=[cpu_time out{i}{j}.time*ones(1,length(index))];
                        else
                            cpu_time=[cpu_time out{i}{j}.time(index)];
                        end
                        Pred_Error=[Pred_Error out{i}{j}.L_pred_valid(index)];
                    end
                end
            else
                if isfield(out{i},'time')
                    N_params=[N_params out{i}.N_params(index)];
                    if length(out{i}.time)==1
                        cpu_time=[cpu_time out{i}.time*ones(1,length(index))];
                    else
                        cpu_time=[cpu_time out{i}.time(index)];
                    end
                    Pred_Error=[Pred_Error out{i}.L_pred_valid(index)];
                end
            end
        end
    end
    N=sort(unique(N_params),'ascend');
    for n=1:length(N)
        ii=find(N(n)==N_params);
        T(n)=mean(cpu_time(ii)); T_std(n)=std(cpu_time(ii));
        E(n)=mean(Pred_Error(ii)); E_std(n)=std(Pred_Error(ii));
    end
end

