function [] = OPA_ANN_Predictions(data_file_name,N_ann)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
load(['Output/' data_file_name '_results.mat']);
N_anneal=N_ann;
for e=1:length(reg_param)
    e
    for ind_K=1:numel(C_fin_en)
        ind_K
        clear pi_KH_full
        for n=1:N_markov
            pi_KH_full{n}=gamma_fin_full{ind_K}(:,n:N_markov:size(gamma_fin_full{ind_K},2),e);
        end
        for n_steps=1:length(N_steps)
            C_pred=C_fin_full{ind_K}(:,:,e);
            [ mean_err_opa_ann_interm(ind_K,n_steps,e)] = Parallel_OPA_ANN_EstimationAndPrediction_v2(pi_KH_full, C_pred,N_steps(n_steps),Y,fraction,1,N_anneal);
        end
    end
end

for ind_K=1:numel(C_fin_en)
    for n_steps=1:length(N_steps)
        mean_err_opa_ann(ind_K,n_steps)=min(squeeze(mean_err_opa_ann_interm(ind_K,n_steps,:)));
    end
end
[XX,YY]=meshgrid(K,N_steps);
figure;surf(XX,YY,mean_err_KMeans','FaceColor','r','Edgecolor','r');hold on;
surf(XX,YY,mean_err_enf','FaceColor','g','Edgecolor','g');surf(XX,YY,mean_err_full','FaceColor','b','Edgecolor','b');
surf(XX,YY,mean_err_opa_ann','FaceColor','y','Edgecolor','b');alpha(0.3)
legend('Kmeans+Markov','Kmeans+ANN','OPA+Markov','OPA+ANN');
 set(gca,'FontSize',20,'LineWidth',2,'ZScale','log');
 xlabel('discretization dimension (number of clusters)','FontSize',14);
 ylabel('depth of prediction (in time steps)','FontSize',14);
 zlabel('mean Euclidean error of prediction','FontSize',14);
 box on
save(['Output/' data_file_name '_results_with_OPA_ANN.mat']);
end

