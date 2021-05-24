function [ x, it, hess_mult ] = spgqp_vec(A, b, x0, normA, my_eps, max_it)

% some magic parameters
m = 7; % the size of generalized Armijo rule buffer
gamma = 0.3; % parameter of Armijo rule
sigma2 = 0.99; % safeguarding parameter

% initialize counters
hess_mult = 0;
it = 0;

x = projection_simplexes(x0); % x \in \Omega
g = A*x - b; hess_mult = hess_mult + 1;
f = get_function_value( x, g, b);

% initialize Armijo buffer
fs = kron(f,ones(m,1));

% Barzilai-Borwein step-size
alpha_bar = 0.95/normA; % 0 < alpha_bar <= 2*norm(inv(A));
alpha_bb = alpha_bar;

gp = get_projected_gradient(x,g,alpha_bar);

while my_stopping_criterion(x,norm2_vec(gp),my_eps) && it < max_it
    
    d = projection_simplexes(x-bsxfun(@times,alpha_bb,g)) - x;

    Ad = A*d; hess_mult = hess_mult + 1;
    dd = dot(d,d); % matrix dot !
    dAd = dot(Ad,d); % matrix dot !
    
    f_max = max(fs);
    
    xi = (f_max - f)./dAd;
    beta_bar = -dot(g,d)./dAd;
    
    beta_hat = gamma*beta_bar + sqrt(gamma^2*beta_bar.^2 + 2*xi);
    
    beta = min(beta_hat,sigma2);
    
    x = x + bsxfun(@times,beta,d);
    g = g + bsxfun(@times,beta,Ad);
    f = get_function_value( x, g, b);
    
    fs(1:end-1,:) = fs(2:end,:);
    fs(end,:) = f;
    
    alpha_bb = dd./dAd;
    
    gp = get_projected_gradient(x,g,alpha_bar);
    
    it = it + 1;

end

end

function [result] = my_stopping_criterion(x,norm_gp,my_eps)
    result = sum(norm_gp > my_eps);
end

function gp = get_projected_gradient(x,g,alpha_bar)
    gp = 1/alpha_bar*(x - projection_simplexes(x - alpha_bar*g));
end

function [ fx ] = get_function_value( x, g, b)
% compute value of quadratic function using gradient
fx = 1/2*dot(g-b,x);
end

function [out] = norm2_vec(x)
    out = sqrt(sum(x.^2,1));
end