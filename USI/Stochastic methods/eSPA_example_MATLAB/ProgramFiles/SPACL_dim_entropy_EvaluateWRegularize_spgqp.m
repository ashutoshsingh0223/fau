function [W]=SPACL_dim_entropy_EvaluateWRegularize_spgqp(X,gamma,C,d,T,W,eps_C,flag_spgqp)
alpha=(X-C*gamma);alpha=sum(alpha.^2,2)';
eps1=1/T;eps2=eps_C;
if flag_spgqp~=1
%options=optimset('GradObj','on','Algorithm','trust-region','MaxIter',20,'Display','off','TolFun',1e-13,'TolCon',1e-13,'TolX',1e-13,...
%    'TolConSQP',1e-13,'TolGradCon',1e-13,'TolPCG',1e-13);
options = optimoptions('fmincon','Algorithm','interior-point','SpecifyObjectiveGradient',true,...
    'Display','off','TolFun',1e-13,'TolCon',1e-13,'TolX',1e-13,'TolConSQP',1e-13,'TolGradCon',1e-13,'TolPCG',1e-13);
A=-eye(d);b=zeros(d,1);
Aeq=ones(1,d);beq=1;
%tic;
[W,fff,flag,output,~,~,hess] =  fmincon(@(x)LogLik_SPACL_W...
    (x,alpha,d,eps1,eps2)...
    ,W,(A),(b),Aeq,beq,[],[],[],options);%toc
%fun=LogLik_SPACL_Lambda(xxx0,gamma,pi,m,K)-LogLik_SPACL_Lambda(xxx,gamma,pi,m,K);[fun]=LogLik_SPACL_gamma(xxx0,X(:,t),pi(:,t),Lambda,eps1,eps2,C,CTC)-fff
else
    CTC = 0.5*eps2*(1./W);alpha_bar =max(1e-12,.5*eps2*min(W));
    % gradient of objective function
    mygrad = 2*CTC.*W + (eps1*alpha+eps2*log(max(W,1e-12)));
   % Dostal-Schoberl step
    W = projection_simplex((W - alpha_bar*mygrad)',d)';

end
end

function [fun,grad]=LogLik_SPACL_W(W,alpha,d,eps1,eps2)
fun=eps1*sum(W.*alpha)+eps2*sum(W.*(log(max(W,1e-12))));
grad=eps1*alpha+eps2.*(log(max(W,1e-12))+ones(1,d));
%H=eps2*(1./max(W,1e-12));
end
