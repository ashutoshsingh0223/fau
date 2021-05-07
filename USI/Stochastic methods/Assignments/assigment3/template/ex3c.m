figure
func = constrainedPlot();


function func = constrainedPlot(p)
    func = @(x1,x2) x1*x2;
    norm_1 = @(x1,x2) power(power(abs(x1),1) + power(abs(x2),1), 1)
    norm_2 = @(x1,x2) power(power(abs(x1),2) + power(abs(x2),2), 1/2)
    norm_inf = @(x1,x2) max(abs(x1), abs(x2))

    hold on
    fc_1 = fcontour(norm_1, [0 2 0 2]);
    fc_1.LineColor = 'r';
    fc_1.LevelList = [1, 1];
    scatter (0.5, 0.5, 'black')
    
    fc_2 = fcontour(norm_2, [0 2 0 2]);
    fc_2.LineColor = 'b';
    fc_2.LevelList = [1, 1];
    scatter (1/sqrt(2), 1/sqrt(2), 'black')
    
    fc_inf = fcontour(norm_inf, [0 2 0 2]);
    fc_inf.LineColor = 'y';
    fc_inf.LevelList = [1, 1];
    scatter (1, 1, 'black')
    
    fc_func = fcontour(func, [0 2 0 2]);
    fc_func.LineColor = 'g';
    fc_func.LevelStep = 0.05;    
    
    grid on
    hold off
end