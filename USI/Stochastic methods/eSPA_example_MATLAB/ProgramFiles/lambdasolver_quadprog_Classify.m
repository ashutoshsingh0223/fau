function [Lambda, timers, lm, output] = lambdasolver_quadprog_Classify( gamma,ppi )
%
% INPUT:
%  gamma - cell of length Ng, each element is matrix K,T
%  ppi - cell of length Ng, each element is matrix m,T
%
% OUTPUT:
%  Lambda - primal solution (matrix of size m,K)
%  timers - time measurement (.assemble_cov, .assemble_mat, .solve_quadprog)
%  lm - Lagrange multipliers, i.e. dual solution
%

% get size of the problem
[ Ng,K,T ] = get_size_of_gamma( gamma );
[ Ng,m,T ] = get_size_of_ppi( ppi );

% todo: check the consistency of Ng,T in gamma and ppi?

% define parameters of quadprog solver
clear options
options.Algorithm = 'interior-point-convex';
options.Display = 'none';%'iter';
options.TolFun = 1e-12;
options.TolCon = 1e-12;
options.TolX = 1e-12;

% prepare covariance blocks
tic
gtt_block = zeros(K,K); % Ahat in derivation
gtpi_block = zeros(K,m); % Bhat in derivation
for g=1:Ng
    gtt_block = gtt_block + gamma{g}*gamma{g}';
    gtpi_block = gtpi_block + gamma{g}*ppi{g}';
end
timers.assemble_cov = toc; % time measurement

% there are some issues with the symmetricity of gtt_block 
% (probably error in multiplication of matrices)
gtt_block = (gtt_block + gtt_block')/2;

tic
% from blocks to hessian matrix and linear-term vector
H = kron(speye(m),gtt_block);
b = -2*reshape(gtpi_block,K*m,1);

% equality constraints (sum of Lambda in rows is 1)
BE = kron(ones(1,m),speye(K));
cE = ones(K,1);

% lower bound
lb = zeros(K*m,1);

% upper bound - not necessary, but maybe it helps to solve problem faster (?)
%ub = [];
ub = ones(K*m,1);

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
Lambda = reshape(lambda_vec,K,m)';

lm.lower = reshape(lm.lower,K,m)';
lm.upper = reshape(lm.upper,K,m)';

end

