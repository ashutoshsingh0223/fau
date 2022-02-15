function y = spline_b2(x)

y = (((-1.5 < x) & (x <= -0.5)) .* (0.5*(x.^2 + 3*x + 2.25))) + ...
    (((-0.5 < x) & (x <=  0.5)) .* (0.5*(-2*x.^2 + 1.5))) + ...
    ((( 0.5 < x) & (x <=  1.5)) .* (0.5*(x.^2 - 3*x + 2.25)));

