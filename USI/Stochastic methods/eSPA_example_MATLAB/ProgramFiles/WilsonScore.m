function [p0,pp,pm] = WilsonScore(p,n)
%UNTITLED Summary of this function goes here
%   http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
%   Wilson score interval with continuity correction
z=1.96;
p0=(p+0.5*z^2/n)./(1+z^2/n);
for i=1:length(p0)
    pm(i)=max(1e-12,(2*n*p(i)+z^2-(z*sqrt(z^2-1/n+4*n*p(i)*(1-p(i))+4*p(i)-2)+1)) / (2*(n+z^2)));
    pp(i)=min(1,(2*n*p(i)+z^2+(z*sqrt(z^2-1/n+4*n*p(i)*(1-p(i))+4*p(i)-2)+1)) / (2*(n+z^2)));
end
end

