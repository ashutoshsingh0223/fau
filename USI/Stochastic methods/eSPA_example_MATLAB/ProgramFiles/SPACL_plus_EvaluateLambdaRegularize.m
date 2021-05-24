function [Lambda]=SPACL_plus_EvaluateLambdaRegularize(pi,gamma,m,K,Lambda);
options=optimset('GradObj','on','Algorithm','interior-point','MaxIter',500,'Display','off','TolFun',1e-13,'TolCon',1e-13,'TolX',1e-13,...
    'TolConSQP',1e-13,'TolGradCon',1e-13,'TolPCG',1e-13);
%options = optimoptions('fmincon','Algorithm','sqp','GradObj','off','CheckGradients',true,'Display','off','TolFun',1e-13,'TolCon',1e-13,'TolX',1e-13,'TolConSQP',1e-13,'TolGradCon',1e-13,'TolPCG',1e-13);
A=-eye(m*K);b=zeros(m*K,1);
Aeq=zeros(K,m*K);beq=ones(K,1);
for k=1:K
    Aeq(k,(k-1)*m+1:k*m)=ones(1,m);
end
%eps1=-reg_param/T/m;
xxx0=reshape(Lambda,K*m,1);
Lambda_prev=Lambda;
%tic;
[xxx,fff,flag,output] =  fmincon(@(x)LogLik_SPACL_Lambda...
    (x,gamma,pi,m,K)...
    ,xxx0,(A),(b),Aeq,beq,[],[],[],options);%toc
%fun=LogLik_SPACL_Lambda(xxx0,gamma,pi,m,K)-LogLik_SPACL_Lambda(xxx,gamma,pi,m,K);[fun]=LogLik_SPACL_gamma(xxx0,X(:,t),pi(:,t),Lambda,eps1,eps2,C,CTC)-fff
[fun]=LogLik_SPACL_Lambda(xxx0,gamma,pi,m,K);
if fun<fff
    Lambda=Lambda_prev;
else
    Lambda=reshape(xxx,m,K);
end
end

function [fun,grad]=LogLik_SPACL_Lambda(Lambda,gamma,pi,m,K)
Lambda=reshape(Lambda,m,K);
fun=-sum(sum(pi.*(log(max(Lambda*gamma,1e-12)))));
grad=-(pi./max(Lambda*gamma,1e-12))*gamma';
end