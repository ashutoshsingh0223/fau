function [params, rmse] = levenbergMarquardt(jacobian_func, residual_func, x, y, lambda)
%   Method implements Levenberg Marquardt method 
%   This method takes in a function `jacobian_func` to calculate the Jacobian at
%   specifc param values.
%   This method takes in a function `residual_func` to calculate the residual at
%   specifc param values.
%   It also takes in initial lambda value
params = [0.994683895702133 0.454516049858223];
delta = 1.0e-6;
error = inf;
count = 0;
%  Run a loop while squared error is greater than delta.
while count < 10000

    %  Calculating Jacobian at inital values of parameters
    A = jacobian_func(params(1), params(2), x);
    %   Calculating transpose(A) * A    
    A_T = transpose(A);
    A_T_A = A_T  * A;
    % Evaluating r at inital values of parameters  
    r_x = residual_func(params(1), params(2), x, y);
    % Calculate direction  of update
    v = -1 * inv(A_T_A + lambda * diag(diag(A_T_A))) * A_T * transpose(r_x);
    % Update x_val(i.e x_i)
    params = params + transpose(v);
    % Calculate squared error.
    residual = residual_func(params(1), params(2), x, y);
    residual_norm = norm(residual);
    rmse = sqrt((residual_norm^2)/length(residual));
    count = count + 1;
%     if mod(count, 1000) == 0
%         v
%         error
%         params
%     end
end


end

