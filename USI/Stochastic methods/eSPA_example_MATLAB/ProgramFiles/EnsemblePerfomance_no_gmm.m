function [Perf,model_label,E]=EnsemblePerfomance_no_gmm(out_DeepNN,out_kmeans,out_pca,out_spacl,out_spacl_gauge,out_svm,out_glm,out_lda,flag_AUC);

[T,N]=size(out_spacl);
    if flag_AUC==1
        c=-1;
    else
        c=1;
    end

for t=1:T
   for n=1:N
       E(n,1,t)=max(c*out_DeepNN{t,n}.L_pred_valid);
       model_label{1}='Deep Learning';
       vv1=[];vv2=[];
       for kk=1:numel(out_kmeans{t,n})
           vv1(kk)=max(c*out_kmeans{t,n}{kk}.L_pred_valid);
           vv2(kk)=out_kmeans{t,n}{kk}.L_pred_valid_Markov;
       end
       E(n,2,t)=max(vv1);
       E(n,3,t)=max(c*vv2);
       model_label{2}='Kmeans+NN';
       model_label{3}='Kmeans+Bayes';
       vv1=[];
       for kk=1:(numel(out_pca{t,n})-1)
           vv1(kk)=max(c*out_pca{t,n}{kk}.L_pred_valid);
       end
       E(n,4,t)=max(vv1);
       model_label{4}='PCA+NN';
       E(n,5,t)=max(c*out_pca{t,n}{kk+1}.L_pred_valid);
       model_label{5}='NN';
       vv1=[];
       for kk=1:(numel(out_spacl{t,n}))
           vv1(kk)=out_spacl{t,n}{kk}.L_pred_valid_Markov;
       end
       E(n,6,t)=max(c*vv1);
       model_label{6}='eSPA';
       vv1=[];
       for kk=1:(numel(out_spacl_gauge{t,n}))
           vv1(kk)=out_spacl_gauge{t,n}{kk}.L_pred_valid_Markov;
       end
       E(n,7,t)=max(c*vv1);
       model_label{7}='goSPA';
       E(n,8,t)=c*out_svm{t,n}.L_pred_valid;
       model_label{8}='SVM';
       E(n,9,t)=out_glm{t,n}.L_pred_valid;
       model_label{9}='GLM';
       E(n,10,t)=out_lda{t,n}.L_pred_valid;
       model_label{10}='LDA';
   best=max(squeeze(E(n,:,t)));
   Perf(n,:,t)=E(n,:,t)-best;
   end
end

end

