function LogLik = LogLikGMM_OfReiduals(x,gamma)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
[K,T]=size(gamma);
D=size(x,1);
for k=1:K
    ind=find(gamma(k,:)==1);
    if length(ind)>1
        mu(:,k)=mean(x(:,ind)')';
        sigma(:,:,k)=cov(x(:,ind)');
        if det(sigma(:,:,k))<1e-12
            sigma(:,:,k)=diag(max(diag(sigma(:,:,k)),1e-12));
        end
    else
        mu(:,k)=x(:,ind);
        sigma(:,:,k)=(1e-12)*eye(D);
    end
    p(k)=sum(ind)/T;
end
gm = gmdistribution(mu',sigma,p);
LogLik=sum(log(max(1e-12,gm.pdf(x'))));
end

