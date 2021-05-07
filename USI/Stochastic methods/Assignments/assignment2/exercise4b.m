function exercise4b()
    error_array_trap = double.empty(0, 10);
    error_array_simp = double.empty(0, 10);
    real_result = (25 * pi) - 5;
    y_axis = single.empty(0, 10);
    for j = 1: 10
        result = quadrature(pi/2, 3*pi, 2 ^ j, "trap");
        result_simp = quadrature(pi/2, 3*pi, 2 ^ j, "simp");
        error_array_trap(j) = abs(result - real_result);
        error_array_simp(j) = abs(result_simp - real_result);
        y_axis(j) = 2 ^ j;
        
    end
    semilogy(error_array_trap,y_axis,error_array_simp, y_axis)
    ylabel('log n')
    legend('Trapezoidal Error','Simpson Error','Location','northwest')
end


function [f_x] = f(x)
    f_x = 4 + (5 * x * sin(x));
%     f_x = (x^2)*(exp(x));
end


function [result] =  quadrature(a, b, n, flag)
    result = 0.0;
%     Computing window size
    h = 1/n;
%     Initializing starting limit
    x_0 = a;
    if flag == "trap"
        for i = 1: n
%             Get upper limit
            x_1 = x_0 + h;
%             Calculate width of trapezoid
            f_sum = f(x_0) + f(x_1);
%             Calculate height of trapezoid
            height = x_1 - x_0;
%             Caculate Area
            area = f_sum * height / 2;
%             Add to composite result
            result = result + area;
%             Step-up lower limit
            x_0 = x_0 + h;
        end
    elseif flag == "simp"
        for i = 1: n/2
            x2i_2 = a + (2*i-2)*h;
            x2i_1 = a + (2*i-1)*h;
            x2i = a + (2*i)*h;
            inner_result = (h * (f(x2i_2) + 4* f(x2i_1) + f(x2i))) / 3;
            result = result + inner_result;
        end
            
    end
 
end