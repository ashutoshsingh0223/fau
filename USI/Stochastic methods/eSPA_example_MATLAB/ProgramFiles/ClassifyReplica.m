function [out_DeepNN,out_kmeans,out_pca,out_spacl,out_gmm,out_svm,out_glm,out_lda]=ClassifyReplica(in)
        tic;
        [out_glm] = LogitRegressionCross_v2(in.X_train,in.pi_train,in.X_valid,in.pi_valid);
        out_glm.time=toc;
        [out_lda] = LDA(in.X_train,in.pi_train,in.X_valid,in.pi_valid);
        out_DeepNN= DeepNN_Classify(in.X_train,in.pi_train, in.X_valid, in.pi_valid,in.flag_AUC);
        out_svm= SVM_Classify_pure(in.X_train, in.pi_train, in.X_valid, in.pi_valid,in.flag_AUC);
%        if size(in.X_train,1)>10 
%         TTT_train=size(in.X_train,2);   TTT_valid=size(in.X_valid,2);
%         XX=MyPCA([in.X_train in.X_valid],10); 
%         [out_glm] = LogitRegressionCross(XX(:,1:TTT_train),in.pi_train,XX(:,1+TTT_train:TTT_train+TTT_valid),in.pi_valid);   
%        else
       % [out_glm] = LogitRegressionCross(in.X_train,in.pi_train,in.X_valid,in.pi_valid);
%        end
        K=in.K;K_means=in.K_means;
        for ind_k=1:length(K)
            K(ind_k)
            out_kmeans{ind_k}= KMeans_Classify(in.X_train,in.pi_train,K(ind_k), in.X_valid, in.pi_valid,in.N_anneal,in.flag_AUC);
            out_pca{ind_k}= PCA_Classify(in.X_train,in.pi_train,K(ind_k), in.X_valid, in.pi_valid,in.N_anneal,in.flag_AUC);
            %out_opd{ind_K,n_ens} = ConvMeansParallelRegularize_Classify_GO_v2(X_train,pi_train,...
            %12    K(ind_K),N_anneal,out_kmeans{ind_K,n_ens}.gamma,reg_param,X_valid,pi_valid,N_neurons,GO_weights);
            out_spacl{ind_k} = SPACL_kmeans_dim_entropy_analytic_v2(in.X_train,in.pi_train,...
                K(ind_k),in.N_anneal,out_kmeans{ind_k},in.reg_param,in.X_valid,in.pi_valid,in.N_neurons,in.flag_AUC);
            %         out_spacl_plus{ind_k} = SPACL_plus_dim_entropy(in.X_train,in.pi_train,...
            %           K(ind_k),in.N_anneal,out_spacl{ind_k},in.reg_param,in.X_valid,in.pi_valid,in.N_neurons,in.flag_AUC);
            if size(in.X_train,1)<size(in.X_train,2)
                out_gmm{ind_k}= GMM_Classify(in.X_train,in.pi_train,K(ind_k), in.X_valid, in.pi_valid,in.N_anneal,in.flag_AUC);
            else
                out_gmm{ind_k}.L_pred_valid = 0*out_kmeans{ind_k}.L_pred_valid;
                out_gmm{ind_k}.time = out_kmeans{ind_k}.time;
            end
            %L_opd(ind_K)=L_opd(ind_K)+1/N_ens*out_opd{ind_K,n_ens}.L_pred_valid;
            %L_opd_mark(ind_K)=L_opd_mark(ind_K)+1/N_ens*out_opd{ind_K,n_ens}.L_pred_Markov;
        end
        out_pca{ind_k+1}= PCA_Classify(in.X_train,in.pi_train,size(in.X_valid,1), in.X_valid, in.pi_valid,in.N_anneal,in.flag_AUC);
        for ind_kmeans=1:length(K_means)
            ind_k=ind_k+1;
            out_kmeans{ind_k}= KMeans_Classify(in.X_train,in.pi_train,K_means(ind_kmeans), in.X_valid, in.pi_valid,in.N_anneal,in.flag_AUC);
        end
end