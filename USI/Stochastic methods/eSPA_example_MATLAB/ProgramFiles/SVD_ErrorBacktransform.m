function [M_out] = SVD_ErrorBacktransform(M,T,Mem,Dim,S)
K=size(M,2);d_full=size(S,1);
M_out=zeros(size(M));
for mem=1:length(Mem)
    M(mem,:)=sqrt(M(mem,:)*((T-Mem(mem))*Dim));
    for k=1:K
        X_red=randn(Dim,(T-Mem(mem)))*M(mem,k);
        X_rec=S*X_red;
        M_out(mem,k)=sqrt(sum(sum(X_rec.^2)))/(d_full*(T-Mem(mem)));
    end
end

