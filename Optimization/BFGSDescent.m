function [xmin] = BFGSDescent(f, x0, eps, verbose)
%BFGSDESCENT Find minimal point of function using inverse BFGS update
%formula

%% Purpose:
% Find xmin to satisfy norm(gradient)<=eps
% Iteration: x_k = x_k + t_k * d_k
% d_k is the BFGS direction. If a descent direction check fails, d_k is set to steepest descent and the inverse BFGS matrix is reset.
% t_k results from Wolfe-Powell

%% Input Definition:
% f: function handle of type [value, gradient] = f(x).
% x0: column vector in R^n (domain point), starting point.
% eps: positive value, tolerance for termination. Default value: 1.0e-3.
% verbose: bool, if set to true, verbose information is displayed

%% Output Definition:
% xmin: column vector in R^n (domain point, box constraints), satisfies norm(gradient)<=eps

%% Required files:
% [t] = WolfePowellSearch(f, x, d, sigma, rho)
%

%% Test cases:
% [xmin]=BFGSDescent(@(x)nonlinearObjective(x), [-0.01;0.01], 1.0e-6, true);
% should return
% xmin close to [0.26;-0.21] with the inverse BFGS matrix being close to
%   0.0078    0.0005
%   0.0005    0.0080
%
% [xmin]=BFGSDescent(@(x)nonlinearObjective(x), [0.6;-0.6], 1.0e-3, true);
% should return
% xmin close to [-0.26;0.21] with the inverse BFGS matrix being close to
%    0.0150    0.0012
%    0.0012    0.0156
%
% [xmin]=BFGSDescent(@(x)bananaValleyObjective(x),[0;1], 1.0e-6, true);
% should return
% xmin close to [1;1] in less than 100 iterations (steepest descent needs about 15.000 iterations). If you have too much
% iterations, then B is maybe not updated properly or you swith to steepest descent to much.
% The inverse BFGS matrix should be close to
%    0.4996    0.9993
%    0.9993    2.0040 (almost singular)

%% Input verification:

try
    x=x0;
    [value, gradient] = f(x);
catch
    error('evaluation of function handle failed!');
end

if (eps <= 0)
    error('range of eps is wrong!');
end

if nargin < 4
    verbose = false;
end

%% Implementation:
% Hints:
% 1. Whenever x changes, you need to update related variables properly!
% 2. eye generates a unit matrix.
% 3. Keep track of the iterations with
% if verbose
%   countIter=countIter+1;
% end

if verbose
    disp('Start BFGSDescent...');
    countIter = 0;
end

%static
sigma=1.0e-3;
rho=1.0e-2;
n=length(x0);
E=eye(n);

%dynamic
B=E;
x_k = x0;
[value, gradient] = f(x_k);
while norm(gradient) > eps
    d_k = -1 * B * gradient;
    if (gradient' * d_k >= 0)
%       descent direction check failed
%       Setting d to steepest descent
        d_k = -gradient;
    end
    [t_k] = WolfePowellSearch(f, x_k, d_k, 1.0e-3, 1.0e-2, true);
    [value_plus, gradient_plus] = f(x_k + t_k * d_k);
    del_gk = gradient_plus - gradient;
    del_x_k = t_k * d_k;
    x_k = x_k + t_k * d_k;
    
    r_k = del_x_k - B * del_gk;
    B_plus = B + (((r_k * del_x_k') + (del_x_k * r_k')) / (del_gk' * del_x_k)) - (((r_k' * del_gk) * (del_x_k * del_x_k')) / (del_gk' * del_x_k)^2);
    
    value = value_plus;
    B = B_plus;
    gradient = gradient_plus;
    countIter = countIter + 1;
    
end

xmin = x_k;


if verbose
    [value, gradient] = f(xmin);
    disp(sprintf('BFGSDescent terminated after %i steps with norm of gradient =%d and the inverse BFGS matrix is\n',countIter, norm(gradient)));
    disp(B);
end

end