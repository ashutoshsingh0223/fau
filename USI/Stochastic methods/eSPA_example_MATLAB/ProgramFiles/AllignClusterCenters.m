function [C2,ind_perm]=AllignClusterCenters(C1,C2)
K=size(C1,2);
dist=zeros(K);
for ind_K1=1:K
    for ind_K2=1:K
        dist(ind_K1,ind_K2)=norm(C1(:,ind_K1)-C2(:,ind_K2));
    end
end

K=size(dist);
ind_i_elim=1:K;ind_j_elim=1:K;
ind_i=[];ind_j=[];
for k=1:K
    [~,i,j]=min_matrix(dist);
    ind_i=[ind_i ind_i_elim(i(1))];
    ind_j=[ind_j ind_j_elim(j(1))];
    ind_i_elim=setdiff(ind_i_elim,ind_i_elim(i(1)));
    ind_j_elim=setdiff(ind_j_elim,ind_j_elim(j(1)));
    dist=dist(setdiff(1:size(dist,2),i(1)),:);
    dist=dist(:,setdiff(1:size(dist,2),j(1)));
end
[~,ind_perm]=sort(ind_i,'ascend');
C2=C2(:,ind_j(ind_perm));
ind_perm=ind_j(ind_perm);

end
