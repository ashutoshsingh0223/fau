mu = [0 0];
sigma = [9, 0; 0, 1];

N = [20, 50, 100, 200, 500, 3000, 10^4, 10^5, 10^6, 10^7, 10^8];

mean_diff = double.empty(0, length(N));
cov_diff = double.empty(0, length(N));


for i = 1:length(N)
X = mvnrnd(mu, sigma, N(i));
[mean_array, covariance, correlation] = estVec(transpose(X));
m_diff = mean_array - mu;
c_diff = covariance - sigma;
mean_diff(i) = norm(m_diff, 2);
cov_diff(i) = norm(c_diff, 2);
end

semilogy(mean_diff,N,cov_diff, N)
ylabel('N')
legend('error in mean','error in cov','Location','northwest')


function [mean_array, covariance, correlation] = estVec(X)
mean_array = mean(X, 2);
covariance = cov(transpose(X));
correlation = corrcoef(transpose(X));
end