rng('default')

%  I am choosing alpha values from 0 - unregularized to 
alpha_values = [0 0.000001 0.00001 0.0001 0.001 0.01 0.025 0.05 0.075 0.1];
residual_norms = zeros(1, length(alpha_values));
error_norms = zeros(1, length(alpha_values));

for j = 1: length(alpha_values)
alpha = alpha_values(j);
n = 100;
x_exact = rand(n,1);

J = 1:n;
J = J(ones(n,1),:);
I = J';
E = ones(n,n);
H = E./(I+J-1);

b = H * x_exact;

[x_sol, residual_norm] = leastSquaresRegularized(H ,b, alpha);
error_norms(j) = norm(x_sol - x_exact);
residual_norms(j) = residual_norm;
end

figure;
plot(alpha_values, residual_norms);
ax = gca;
ax.XGrid = 'on';
hold off;

title('Plot of residual norms w.r.t alpha');
xlabel('alpha');
ylabel('residual norm');


figure;
plot(alpha_values, error_norms);
ax = gca;
ax.XGrid = 'on';
hold off;

title('Plot of error norms w.r.t alpha');
xlabel('alpha');
ylabel('error norm'); 



function [sol, residual_norm] = leastSquaresRegularized(A, b, alpha)

% LEASTSQUARES Summary of this function goes here
% This method takes in a matrix `A` and vector `b` and 
% returns least sqaures solution `sol`
% euclidean norm of the residual `residual_norm`

%  alpha is the resularization parameter

% Calculation A_transpose*A and A_transpose*b
A_T_A = transpose(A) * A + alpha;
A_T_b = transpose(A) * b;

[m_, s] = size(A);
sol = zeros(s,1);

for j = 1:(s-1)
    for i = s:-1:j+1
        if A_T_A(j,j) ~= 0
            m = A_T_A(i,j) / A_T_A(j,j);
            A_T_A(i,:) = A_T_A(i,:) - m * A_T_A(j,:);
            A_T_b(i) = A_T_b(i) - m * A_T_b(j);
        end
    end
end 
% Calculation x_m (Last element in the solution vector)
if A_T_A(s,s) ~= 0
    sol(s) = A_T_b(s)/A_T_A(s,s);
end

% Backward substitution
for i = s-1:-1:1                    
    sum_ = 0;
    for j = s:-1:i+1                
        sum_ = sum_ + A_T_A(i,j)*sol(j);    
    end
    if A_T_A(i,i) ~= 0
        sol(i) = (A_T_b(i)- sum_)/A_T_A(i,i);
    end
end

residual = b - A * sol;
residual_norm = sqrt(sum(residual.^2));

end


