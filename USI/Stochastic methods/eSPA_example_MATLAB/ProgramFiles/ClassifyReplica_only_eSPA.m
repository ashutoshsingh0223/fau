function [out_spacl,out_spacl_gauge]=ClassifyReplica_only_eSPA(in)
Gauge_Aneal_Mult_Factor=1;
if in.flag_AUC==1
        K=in.K;K_means=in.K_means;
        for ind_k=1:length(K)
            K(ind_k)
            RG=rng;
            out_spacl{ind_k} = SPACL_kmeans_dim_entropy_analytic_v2(in.X_train,in.pi_train,...
                K(ind_k),in.N_anneal,[],in.reg_param,in.X_valid,in.pi_valid,in.N_neurons,in.flag_AUC);
            out_spacl_gauge{ind_k} = SPACL_kmeans_dim_entropy_analytic_gauge(in.X_train,in.pi_train,...
                K(ind_k),Gauge_Aneal_Mult_Factor*in.N_anneal,[],in.gauge_param,in.X_valid,in.pi_valid,in.N_neurons,in.flag_AUC,RG);
        end
end
