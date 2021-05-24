function [x,y,Z]=ContourLambdaXGamma(Domain_X,Domain_Y,out,flag_mark)

[x,y]=meshgrid(Domain_X,Domain_Y);
d=size(out.C,1);
K=size(out.C,2);
X_valid=zeros(d,prod(size(x)));
k=1;
for i=1:size(x,1)
    for j=1:size(x,2)
        X_valid(1:2,k)=[x(i,j);y(i,j)];
        k=k+1;
    end
end

if ~flag_mark
    [gamma_valid,LLL2]=EvaluateGammaParallel_Lukas_vect(diag(sqrt(out.W))*X_valid,out.C,prod(size(x)),d,K,[],0);
else
    [gamma_valid]=SPACL_EvaluateGamma_valid(diag(sqrt(out.W))*X_valid,out.C,prod(size(x)),K);
end
pi=out.P*gamma_valid;
Z=zeros(size(x,1),size(x,2));
k=1;
for i=1:size(x,1)
    for j=1:size(x,2)
        Z(i,j)=pi(1,k);
        k=k+1;
    end
end

