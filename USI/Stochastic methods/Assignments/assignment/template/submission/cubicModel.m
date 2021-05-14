crude_oil = readtable('~/Desktop/ashutosh/fau/USI/Stochastic methods/Assignments/assignment/data/crudeOil.txt');
kerosene = readtable('~/Desktop/ashutosh/fau/USI/Stochastic methods/Assignments/assignment/data/kerosene.txt');

%  Setting column names for easier access. 'extra' column is where % from
%  change is getting stored when read table is reading it.
crude_oil.Properties.VariableNames = {'Year' 'Production' 'Change' 'extra'};
kerosene.Properties.VariableNames = {'Year' 'Production' 'Change' 'extra'};

%  crude oil production is being read as string. Casting the type to
%  double
crude_oil.Production = str2double(crude_oil.Production);

% Forming A and b for crude oil
A_crude_oil = zeros(length(crude_oil.Production), 4);
A_crude_oil(:, 1) = 1;

% Scaling data for crude oil 
crude_year_scaled_min = crude_oil.Year./min(crude_oil.Year);
crude_prod_scaled = -1 + 2.*(crude_oil.Production - min(crude_oil.Production))./(max(crude_oil.Production) - min(crude_oil.Production));


A_crude_oil(:, 2) = crude_year_scaled_min;
% Adding square of production as another column
A_crude_oil(:, 3) = crude_year_scaled_min.^2;
% Adding cube of production as another column
A_crude_oil(:, 4) = crude_year_scaled_min.^3;
b_crude_oil = crude_prod_scaled;

%  Forming A and b for kerosene
A_kerosene = zeros(length(kerosene.Production), 4);
A_kerosene(:, 1) = 1;

% Scaling data for Kerosene
kerosene_year_scaled_min = kerosene.Year./min(kerosene.Year);
ker_prod_scaled = -1 + 2.*(kerosene.Production - min(kerosene.Production))./(max(kerosene.Production) - min(kerosene.Production));


A_kerosene(:, 2) = kerosene_year_scaled_min;
% Adding square of production as another column
A_kerosene(:, 3) = kerosene_year_scaled_min.^2;
% Adding cube of production as another column
A_kerosene(:, 4) = kerosene_year_scaled_min.^3;
b_kerosene = ker_prod_scaled;


%  Running least squares for crude oil. Till Year 2011
[sol_crude_oil, norm_crude_oil, se_c_o, rmse_c_0] = leastSquares(A_crude_oil(1: end - 1, :),b_crude_oil(1: end-1));

%  Running least squares for kerosene. Till Year 2011
[sol_kerosene, norm_kerosene, se_k, rmse_k] = leastSquares(A_kerosene(1: end - 1, :),b_kerosene(1: end-1));

% Getting predicted values
b_kerosene_predicted = A_kerosene * sol_kerosene;
b_crude_oil_predicted = A_crude_oil * sol_crude_oil;

x = linspace(1, length(b_crude_oil_predicted),length(b_crude_oil_predicted));



figure;
plot(x,b_kerosene_predicted);

hold on;
plot(x, b_kerosene);
hold off;

title('Line Plot of predicted and real production for kerosene-cubic');
xlabel('Years');
ylabel('Production'); 
legend('predicted values','actual values');



figure;
plot(x,b_crude_oil_predicted);
hold on;
plot(x, b_crude_oil);
hold off;

title('Line Plot of predicted and real production for crude oil-cubic');
xlabel('Years'); 
ylabel('Production'); 
legend('predicted values','actual values');

b_crude_predicted_unscaled = (((b_crude_oil_predicted + 1)./2) .* (max(crude_oil.Production) - min(crude_oil.Production))) + min(crude_oil.Production);  

b_kerosene_predicted_unscaled = (((b_kerosene_predicted + 1)./2) .* (max(kerosene.Production) - min(kerosene.Production)))  + min(kerosene.Production);



