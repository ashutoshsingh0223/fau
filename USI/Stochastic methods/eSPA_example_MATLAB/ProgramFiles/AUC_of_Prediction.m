function [AUC] = AUC_of_Prediction(gamma,P,pi_valid,flag_nn,flag_AUC)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if flag_nn==0
    xx=P*gamma;
else
    xx=P(gamma);
end
if flag_AUC==1
    [~,~,~,AUC] = perfcurve(pi_valid(1,:)',xx(1,:)',1);
    AUC=sum(AUC);
    if AUC<0.5
        AUC=1-AUC;
    end
    AUC=-AUC*size(pi_valid,1)*size(pi_valid,2);
else
    AUC=0;
    for t=1:size(xx,2)
        AUC=AUC+KLdivergence(pi_valid(:,t),xx(:,t));
    end
end
end

