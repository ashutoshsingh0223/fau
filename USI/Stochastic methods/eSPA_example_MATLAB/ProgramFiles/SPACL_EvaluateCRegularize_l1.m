function [C]=SPACL_EvaluateCRegularize_l1(C,X,gamma,eps_CL,K,d,H);
%    C=X*gamma'*inv(gamma*gamma'+2*eps_CL/(K*(K-1))*(K*eye(K)-ones(K)));
    %C=((gamma*gamma'+2*eps_CL/(K*(K-1))*(K*eye(K)-ones(K)))\(X*gamma')')';
options=optimset('MaxIter',500000,'Display','off','TolFun',1e-13,'TolX',1e-13,'MaxFunEvals',1e6);
%options = optimoptions('fmincon','Algorithm','sqp','GradObj','off','CheckGradients',true,'Display','off','TolFun',1e-13,'TolCon',1e-13,'TolX',1e-13,'TolConSQP',1e-13,'TolGradCon',1e-13,'TolPCG',1e-13);
xxx0=reshape(C,K*d,1);
%tic;
[xxx,fff,flag,output] =  fminsearch(@(x)SPACL_C_l1...
    (x,X,gamma,d,K,H,eps_CL)...
    ,xxx0,options);%toc
%fun=LogLik_SPACL_Lambda(xxx0,gamma,pi,m,K)-LogLik_SPACL_Lambda(xxx,gamma,pi,m,K);[fun]=LogLik_SPACL_gamma(xxx0,X(:,t),pi(:,t),Lambda,eps1,eps2,C,CTC)-fff
C=reshape(xxx,d,K);

end

function [fun]=SPACL_C_l1(x,X,gamma,d,K,H,eps_C)
C=reshape(x,d,K);
fun=norm(X - C*gamma,'fro')^2+eps_C/(K*(K-1))*sum(abs(H*x));
end