function [Lambda, timers, lm,output] = lambdasolver_label_quadprog( x, ppi, C )
%
% INPUT:
%  x - cell of length Ng, each element is matrix n,T  
%  ppi - cell of length Ng, each element is matrix L,T
%  C - matrix of size n,K
%
% OUTPUT:
%  Lambda - primal solution (matrix of size K,L)
%  timers - time measurement (.assemble_cov, .assemble_mat, .solve_quadprog)
%  lm - Lagrange multipliers, i.e. dual solution
%

% get size of the problem
Ng = 1;
[ n,T ] = size( x );
[ L,T ] = size( ppi );
[ n,K ] = size( C );

% todo: check the consistency of Ng,n,L,K,T?

% define parameters of quadprog solver
clear options
options.Algorithm = 'interior-point-convex';
options.Display = 'none';%'iter';
options.TolFun = 1e-12;
options.TolCon = 1e-12;
options.TolX = 1e-12;

% prepare covariance blocks
tic
Ahat = zeros(L,L);
Bhat = zeros(n,L);
for g=1:Ng
    Ahat = Ahat + ppi*ppi';
    Bhat = Bhat + x*ppi';
end
timers.assemble_cov = toc; % time measurement

tic
% from blocks to hessian matrix and linear-term vector
CI = kron(C,speye(L));
IA = kron(speye(n),Ahat);
H = CI'*IA*CI;

% there are some issues with the symmetricity of Ahat
% (probably error in multiplication of matrices)
H = (H + H')/2;

b = -2*CI'*reshape(Bhat',n*L,1);

% equality constraints (sum of Lambda in rows is 1)
BE = kron(ones(1,K),speye(L));
cE = ones(L,1);

% lower bound
lb = zeros(K*L,1);

% upper bound - not necessary, but maybe it helps to solve problem faster (?)
%ub = [];
ub = ones(K*L,1);

timers.assemble_mat = toc; % time measurement

% solve QP using quadprog
tic
[lambda_vec,~,~,output,lm] = quadprog( ... % [X,FVAL,EXITFLAG,OUTPUT,LAMBDA] = quadprog(
        2*H, ... % Hessian matrix (watch out! quadprog is for 0.5*x'*H*x + b'*x)
        b,  ... % linear term
        [], ... % linear inequalities - matrix
        [], ... % linear inequalities - rhs
        BE, ... % linear equalities - matrix
        cE, ... % linear equalities - rhs
        lb, ... % lower bound
        ub, ... % upper bound
        [], ... % initial
        options); % options
timers.solve_quadprog = toc; % time measurement

% from vec to mat, test it using "a = [1;2;3;4]; reshape(a,2,2)"
Lambda = reshape(lambda_vec,L,K)';

lm.lower = reshape(lm.lower,L,K)';
lm.upper = reshape(lm.upper,L,K)';

end

