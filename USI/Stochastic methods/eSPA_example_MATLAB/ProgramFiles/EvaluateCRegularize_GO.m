function [C,L,norm_C]=EvaluateCRegularize_GO(X,gamma,T,d,K,eps,xt2,H2);
    xt=zeros(d,K);gt=zeros(K);
    for t=1:T
        xt=xt+X(:,t)*gamma(:,t)';
        gt=gt+gamma(:,t)*gamma(:,t)';
    end
        
    xt=xt./(T*d);
    gt=gt./(T*d);
    %C=xt/(gt+eps/(d*K)*eye(K));
    H1=zeros(d*K);H2=eye(d*K);
    b=zeros(d*K,1);
    for i1=1:K
        for i2=1:K
            H1((i1-1)*d+1:i1*d,(i2-1)*d+1:i2*d)=gt(i1,i2)*eye(d);
        end
    b((i1-1)*d+1:i1*d)=-2*xt(:,i1)';   
    end
    H=sparse(2*(H1+eps*H2));
    
  % define parameters of quadprog solver
clear options
options = optimoptions('quadprog','Display','off');

options.Algorithm = 'interior-point-convex';
%options.Display = 'none';%'iter';
options.TolFun = 1e-12;
options.TolCon = 1e-12;
options.TolX = 1e-12;
    [C,fval,~,output,lm] = quadprog( ... % [X,FVAL,EXITFLAG,OUTPUT,LAMBDA] = quadprog(
        H, ... % Hessian matrix (watch out! quadprog is for 0.5*x'*H*x + b'*x)
        b,  ... % linear term
        [], ... % linear inequalities - matrix
        [], ... % linear inequalities - rhs
        [], ... % linear equalities - matrix
        [], ... % linear equalities - rhs
        zeros(d*K,1), ... % lower bound
        [], ... % upper bound
        [], ... % initial
        options); % options

    
    norm_C=C'*H2*C;
    C=reshape(C,d,K);
    %C=xt/(gt+2*eps/(d*(K-1)*K)*(K*eye(K)-ones(K)));
    L=fval+xt2/(T*d);
end
