function ex3a()
    figure
    hold on
%     vars = [1 2 4 6 8 inf]
%     for i = 1:length(vars)
%     f = drawCircle(vars(i));
    func = drawFunc()
%     end
    
end

function func = drawFunc()
    func = @(x1,x2) x1*x2;
%     fcontour(func, [-1 1 -1 1]);
%     func = @(x1,x2) x1*x2;
    fcontour(func, [0 0.25 0 0.25]);
end


function f = drawCircle(p)
    if p == inf
        f = @(x1,x2) max(abs(x1), abs(x2));
        fc = fcontour(f, [-2 2 -2 2]);
        fc.LevelList = [1, 1];
        fc.LineColor = 'r';
    else
         f = @(x1,x2) power(power(abs(x1),p) + power(abs(x2),p), 1/p);
         fc = fcontour(f, [-2 2 -2 2]);
         fc.LevelList = [1, 1];
    end
end