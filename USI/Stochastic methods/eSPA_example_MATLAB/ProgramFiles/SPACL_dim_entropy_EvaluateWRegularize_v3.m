function [W]=SPACL_dim_entropy_EvaluateWRegularize_v3(X,gamma,C,d,T,W,eps_C)
%options=optimset('GradObj','on','Algorithm','sqp','MaxIter',20,'Display','off','TolFun',1e-13,'TolCon',1e-13,'TolX',1e-13,...
%    'TolConSQP',1e-13,'TolGradCon',1e-13,'TolPCG',1e-13,'MaxFunEval',20000,'UseParallel',false);
%C_orig=diag(1./min(sqrt(W),1e-5))*C;
b=sum((X-C*gamma).^2,2);
D1=ones(d,1);
z=b./(T*eps_C)+D1;
%max_arg=max(arg);
W = zeros(1,length(z));
for i=1:length(z)
   W(i) = 1/sum(exp(-z + z(i))); 
end

%W=(exp(-arg+max_arg)./sum(exp(-arg+max_arg)))';
end


