function [gamma]=SPACL_plus_EvaluateGamma(X,pi,C,Lambda,T,K,m,d,reg_param,gamma);
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
options=optimset('GradObj','on','Algorithm','sqp','MaxIter',5,'Display','off','TolFun',1e-13,'TolCon',1e-13,'TolX',1e-13,...
    'TolConSQP',1e-13,'TolGradCon',1e-13,'TolPCG',1e-13);
A=-eye(K);b=zeros(K,1);
Aeq=ones(1,K);beq=1;
eps1=-reg_param/T/m;eps2=1/T/d;CTC=C'*C;mm=min(abs(eps1),abs(eps2));
eps1=eps1/mm;eps2=eps2/mm;

for t=1:T
    xxx0=gamma(:,t);
    [gamma(:,t),fff,flag,output] =  fmincon(@(x)LogLik_SPACL_gamma...
        (x,X(:,t),pi(:,t),Lambda,eps1,eps2,C,CTC)...
        ,xxx0,(A),(b),Aeq,beq,[],[],[],options);
    %[fun]=LogLik_SPACL_gamma(xxx0,X(:,t),pi(:,t),Lambda,eps1,eps2,C,CTC)-fff;
    %if fun<0
    %    keyboard
    %end
end
end

function [fun,grad]=LogLik_SPACL_gamma(gamma,x,pi,Lambda,eps1,eps2,C,CTC)
fun=eps1*pi'*log(max(Lambda*gamma,1e-12))+eps2*norm(x - C*gamma,'fro')^2;
grad=eps1*Lambda'*(pi./max(Lambda*gamma,1e-12))+2*eps2*(CTC*gamma - C'*x);
end