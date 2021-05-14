rng('default');

n_s = [50 100 200 300 400 500 1000];
condition_numbers = zeros(1, length(n_s));
error_norms = zeros(1, length(n_s));

for j = 1: length(n_s) 
    n = n_s(j);
    x_exact = rand(n,1);
    
    J = 1:n;
    J = J(ones(n,1),:);
    I = J';
    E = ones(n,n);
    H = E./(I+J-1);
    
    b = H * x_exact;
    
    [x_sol, residual_norm, se, rmse] = leastSquares(H ,b);
    
    condition_numbers(j) = cond(H);
    error_norms(j) = norm(x_sol - x_exact);
    
end

figure;
plot(n_s, condition_numbers);
hold off;

title('Plot of condition numbers of H w.r.t n');
xlabel('n');
ylabel('condition number');


figure;
plot(n_s, error_norms);
hold off;

title('Plot of error norms w.r.t n');
xlabel('n');
ylabel('error norm'); 

