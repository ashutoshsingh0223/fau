A_1 = [1 0; 1 0; 1 0];
b_1 = [5 2 4];


[sol, norm, se, rmse] = leastSquares(A_1, transpose(b_1));


A_2 = [1 1 0; 0 1 1; 1 2 1; 1 0 1];
b_2 = [2 2 3 4];

[sol, norm, se, rmse] = leastSquares(A_2, transpose(b_2));