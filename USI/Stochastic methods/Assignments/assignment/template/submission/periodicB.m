temperature = readtable('~/Desktop/ashutosh/fau/USI/Stochastic methods/Assignments/assignment/data/temperature.txt');


% Using data from 1960 to 1963
% Since this is 4 years and we have data each month we have a total of 36
% instances

temperature_60_63 = temperature(97:99,1:13);



% Using data from 1960 to 19670
% Since this is 10 years and we have data each month we have a total of 120
% instances

temperature_60_70 = temperature(97:106,1:13);

% Converting matrices to monthy time series. Taking only from second column
% to avoid 'time' column in the table
temperature_60_63_series = reshape(table2array(temperature_60_63(:,2:end))', [], 1);
x = linspace(1, length(temperature_60_63_series),length(temperature_60_63_series));
A_60_63 = zeros(length(temperature_60_63_series), 4);
A_60_63(:, 1) = 1;
A_60_63(:, 2) = cos(2 * pi * x/12);
A_60_63(:, 3) = sin(2 * pi * x/12);
A_60_63(:, 4) = cos(4 * pi * x/12);
b_60_63 = temperature_60_63_series;

[sol, norm, se, rmse] = leastSquares(A_60_63, b_60_63);

b_60_63_pred = A_60_63 * sol;




temperature_60_70_series = reshape(table2array(temperature_60_70(:,2:end))', [], 1);
x_2 = linspace(1, length(temperature_60_70_series),length(temperature_60_70_series));
A_60_70 = zeros(length(temperature_60_70_series), 4);
A_60_70(:, 1) = 1;
A_60_70(:, 2) = cos(2 * pi * x_2/12);
A_60_70(:, 3) = sin(2 * pi * x_2/12);
A_60_70(:, 4) = cos(4 * pi * x_2/12);
b_60_70 = temperature_60_70_series;

[sol_60_70, norm_60_70, se_60_70, rmse_60_70] = leastSquares(A_60_70, b_60_70);

b_60_70_pred = A_60_70 * sol_60_70;

figure;
plot(x,b_60_63_pred);

hold on;
plot(x, b_60_63);
hold off;

title('Line Plot of predicted and real tempertures 1960-1963 - periodicB');
xlabel('Months from 1960-1963');
ylabel('Temperature'); 
legend('predicted values','actual values');



figure;
plot(x_2,b_60_70_pred);

hold on;
plot(x_2, b_60_70);
hold off;

title('Line Plot of predicted and real tempertures 1960-1970 - periodicB');
xlabel('Months from 1960-1970');
ylabel('Temperature'); 
legend('predicted values','actual values');


