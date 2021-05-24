function [ x, it, hess_mult, gp_norms ] = spgqp(A, b, x0, normA, my_eps, max_it)
%SPGQP Spectral Projected Gradient method for simplex-constrained QP
%
% More details:
%  E. G. Birgin, J. M. Martinez, and M. M. Raydan. Nonmonotone spectral
%   projected gradient methods on convex sets. SIAM Journal on Optimization,
%   10:1196?1211, 2000.
%  L. Pospisil. Development of Algorithms for Solving Minimizing Problems with Convex Qua-
%   dratic Function on Special Convex Sets and Applications. PhD thesis, VSB-TU Ostrava,
%   <a href="matlab:web('http://am.vsb.cz/export/sites/am/cs/theses/pospisil_phd.pdf')">online</a>.
%

% some magic parameters
m = 7; % the size of generalized Armijo rule buffer
gamma = 0.3; % parameter of Armijo rule
sigma2 = 0.99; % safeguarding parameter

% initialize counters
hess_mult = 0;
it = 0;

x = projection_simplex(x0); % x \in \Omega
g = A*x - b; hess_mult = hess_mult + 1;
f = get_function_value( x, g, b);

% initialize Armijo buffer
fs = f*ones(m,1);

% Barzilai-Borwein step-size
alpha_bar = 0.95/normA; % 0 < alpha_bar <= 2*norm(inv(A));
alpha_bb = alpha_bar;

gp = get_projected_gradient(x,g,alpha_bar);

gp_norms(1) = norm(gp);

while my_stopping_criterion(x,norm(gp),my_eps) && it < max_it
    
    d = projection_simplex(x-alpha_bb*g) - x;
    
    Ad = A*d; hess_mult = hess_mult + 1;
    dd = dot(d,d);
    dAd = dot(Ad,d);
    
    f_max = max(fs);
    
    xi = (f_max - f)/dAd;
    beta_bar = -dot(g,d)/dAd;
    beta_hat = gamma*beta_bar + sqrt(gamma^2*beta_bar^2 + 2*xi);
    
    beta = min([sigma2,beta_hat]);
    
    x = x + beta*d;
    g = g + beta*Ad;
    f = get_function_value( x, g, b);
    
    fs(1:end-1) = fs(2:end);
    fs(end) = f;
    
    alpha_bb = dd/dAd;
    
    gp = get_projected_gradient(x,g,alpha_bar);
    
    it = it + 1;
    gp_norms(it+1) = norm(gp);

end

end

function [result] = my_stopping_criterion(x,norm_gp,my_eps)
result = (norm_gp > my_eps);
end

function gp = get_projected_gradient(x,g,alpha)
gp = 1/alpha*(x - projection_simplex(x - alpha*g));
end

function [ fx ] = get_function_value( x, g, b)
% compute value of quadratic function using gradient
fx = 1/2*dot(g-b,x);
end
