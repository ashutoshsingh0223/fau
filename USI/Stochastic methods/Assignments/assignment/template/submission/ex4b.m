%% Reading data
% Years
years = [1999 2000 2001 2002 2003 2004 2005 2006];
years_min_scaled = years./1999;
years_scaled = -1 + 2.*(years - min(years))./(max(years) - min(years));


% Consumption array from the data
consumption = [14.09 15.90 16.60 25.17 41.66 47.95 50.33 54.85];
consumption_scaled = -1 + 2.*(consumption - min(consumption))./(max(consumption) - min(consumption));

% Change array from the dataset. Treating this y variable
change = [4.68 12.85 4.40 51.63 65.51 15.10 4.96 8.98];


%% Log linearized model and comparision with RMSE of exponential model

% Forming a blank data matrix of proper dimentions for nuclear dataset.
A = zeros(8, 2);


%% Min scaled year column and no scaling on consumption
% Set first column to be all 1s
A(:, 1) = 1;
% For log linearized model : log(y) = k1 + a2* log(x)
%  Take log of year values and store in our data matrix
A(:, 2) = log(years_min_scaled);

%  Taking log of consumption as b for leastSquares solver
b = log(consumption);
[sol, residual_norm, se, rmse] = leastSquares(A,transpose(b));
% sol is our solution sol = [k1 a2]
%  Calculate y predicted. without log
% for getting `a1` take `exp(k1)`
consumption_predicted = exp(sol(1))*years_min_scaled.^sol(2);

% Calculating residual and then RMSE for exponential model
residual = consumption - consumption_predicted;
norm_exp = sqrt(sum(residual.^2));
se_exp = norm_exp^2;
rmse_exp = sqrt(se_exp/ 2);




%% Levenberg Marquardt for non linear least squares

% Calculating residual as: r = a1*x.^a2 - y
% Put y = change and x=consumption

[result, rmse_lm] = levenbergMarquardt(@Jacobian, @Residual, years_min_scaled, consumption, 5);
% Generating y values from given model, with parama calculated from LM
% algorithm
y = model(result(1), result(2), years_min_scaled);
% [result, rmse_lm] = levenbergMarquardt(@Jacobian, @Residual, 1:8, consumption_scaled, 5);




figure;
plot(years,consumption);

hold on;

plot(years,consumption_predicted);

hold on;
plot(years, y);
hold off;

title('Line Plot of predicted and real nuclear production values');
xlabel('Years');
ylabel('Production of nuclear energy'); 
legend('True', 'Log linearized','Levenberg Marquardt');


function [y] = model(a1, a2, x)
        y = a1*x.^a2;
end

function [r] = Residual(a1, a2, x, y)
%     Function that takes in params and x and y to compute the residual
    r = a1*x.^a2 - y;
end


function [J] = Jacobian(a1, a2, x)
%     Function that takes in x column and param values to return the
%     jacobian of the residual
    J =  transpose([x.^a2; a1*log(x).*(x.^a2)]);
end








