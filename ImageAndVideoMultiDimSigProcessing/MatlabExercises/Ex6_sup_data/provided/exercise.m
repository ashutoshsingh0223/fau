
x =-3:0.01:3;
size(x);
y_0 = zeros(size(x));
for i=1:size(x,2)
    y_0(i) = spline_b0(x(i));
end

subplot(2,2,1)
plot(y_0);
title("zero")

% Plotting further order splines by calculating with numerical convolution
% spline_n = spline_(n-1) *(conv) spline_0
[x_1, y_1] = spline_conv(y_0, y_0);


% Problems the results calculated manually are not centered properly
% althought the values I get are correct.

subplot(2,2,2)
plot(x_1, y_1);
title("one")

[x_2, y_2] = spline_conv(y_1, y_0);

subplot(2,2,3)
plot(x_2, y_2);
title("two")

[x_3, y_3] = spline_conv(y_2, y_0);

subplot(2,2,4)
plot(x_3, y_3);
title("three")




