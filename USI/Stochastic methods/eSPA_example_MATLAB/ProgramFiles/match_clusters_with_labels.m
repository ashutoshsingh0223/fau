function [Match] = match_clusters_with_labels(out,labels,p_tol)

Region_labels={'LDH','RDH','LVH','RVH'};
Resilience_labels={'CTRL','SUS','RES','INT'};
NK=numel(out);NW=length(out{1}.gamma);
for i=1:numel(Region_labels)
    i
    ind_i=find(double(labels.brain_reg==Region_labels{i})==1);
    for j=1:numel(Resilience_labels)
        j
        ind_j=find(double(labels.resilience(ind_i)==Resilience_labels{j})==1)';
        pi_label=zeros(1,length(ind_i));pi_label(ind_j)=1;
        for k=1:NK
            for w=1:NW
                local_gamma=zeros(size(out{k}.gamma{w},1),length(ind_i));
                p=zeros(1,size(out{k}.gamma{w},1));mu_inv=p;pp=mu_inv;pm=mu_inv;
                for ind_k=1:size(out{k}.gamma{w},1)
                    local_gamma(ind_k,:)=out{k}.gamma{w}(ind_k,ind_i)';
                    [tbl{ind_k},~,p(ind_k)] = crosstab(pi_label,local_gamma(ind_k,:));
                    mu_inv(ind_k)=sum(local_gamma(ind_k,ind_j))./length(local_gamma(ind_k,ind_j));
                   [mu_inv(ind_k),pp(ind_k),pm(ind_k)] = WilsonScore(mu_inv(ind_k),length(local_gamma(ind_k,ind_j)));
               end
                [~,ind]=find(p<p_tol);
                [~,mm]=min(p);
                Match{i,j}.n(k,w)=length(ind);
                Match{i,j}.p_min(k,w)=p(mm);
                Match{i,j}.gamma_min{k,w}=local_gamma(mm,:)';
                Match{i,j}.pi_stress{k,w}=pi_label;
                Match{i,j}.tbl{k,w}=tbl{mm};
                Match{i,j}.Lambda{k,w}=tbl{mm};                
                Match{i,j}.mu_inv{k,w}=mu_inv;
                Match{i,j}.pp{k,w}=pp-mu_inv;
                Match{i,j}.pm{k,w}=mu_inv-pm;
                for ii=1:size(tbl{mm},2)
                    Match{i,j}.Lambda{k,w}(:,ii)=Match{i,j}.Lambda{k,w}(:,ii)./sum(Match{i,j}.Lambda{k,w}(:,ii));
                end
            end
        end
    end
end

