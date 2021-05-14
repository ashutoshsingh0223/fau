function [sol, norm, se, rmse] = leastSquares(A,b)
%LEASTSQUARES Summary of this function goes here
% This method takes in a matrix `A` and vector `b` and 
% returns least sqaures solution `sol`
% euclidean norm of the residual `norm`
% SE of the residual `se`
% and RMSE of the solution `rmse`

% Calculation A_transpose*A and A_transpose*b
A_T_A = transpose(A) * A;
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
norm = sqrt(sum(residual.^2));
se = norm^2;

rmse = sqrt(se / m_);

end

