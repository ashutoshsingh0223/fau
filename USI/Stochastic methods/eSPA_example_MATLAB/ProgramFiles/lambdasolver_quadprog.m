function [Lambda, timers, lm, output] = lambdasolver_quadprog( gamma )

%
% INPUT:
%  gamma
%
% OUTPUT:
%  Lambda - primal solution
%  timers - time measurement (.assemble_cov, .assemble_mat, .solve_quadprog)
%  lm - Lagrange multipliers, i.e. dual solution (just for checking KKT later)
%

% get size of the problem
%[ Ng,K,T ] = get_size_of_gamma( gamma );
K=size(gamma,1);T=size(gamma,3);
% define parameters of quadprog solver
clear options
options.Algorithm = 'interior-point-convex';
options.Display = 'none';%'iter';
options.TolFun = 1e-12;
options.TolCon = 1e-12;
options.TolX = 1e-12;

% prepare covariance blocks
tic
gtt_block = zeros(K,K);
gt1t_block = zeros(K,K);
%gt1t1_block = zeros(K,K); % for fun
    for t=1:T
        gtt_block = gtt_block + gamma(:,1,t)*gamma(:,1,t)';
        gt1t_block = gt1t_block + gamma(:,1,t)*gamma(:,2,t)';
%        gt1t1_block = gt1t1_block + gamma{g}(:,t+1)*gamma{g}(:,t+1)';
    end
timers.assemble_cov = toc; % time measurement

tic
% from blocks to hessian matrix and linear-term vector
H = kron(speye(K),gtt_block);
b = -2*reshape(gt1t_block,K*K,1);

% equality constraints (sum of Lambda in each column through rows is 1)
BE = kron(ones(1,K),speye(K));
cE = ones(K,1);

% lower bound
lb = zeros(K*K,1);

% upper bound - not necessary, but maybe it helps to solve problem faster (?)
ub = [];
%ub = ones(K*K,1);

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
Lambda = reshape(lambda_vec,K,K)';

lm.lower = reshape(lm.lower,K,K)';
lm.upper = reshape(lm.upper,K,K)'; % quadprog returns zeros if ub is not provided

end

