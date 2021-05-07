############### Example1: Gibbs Sampler page 80 ##################
set.seed(12345)

# fix setting
n <- 50 # num of observations
mu.true <- 1 # True mean
s2.true <- 0.5 # true variance

M <- 10000 # no. of gibbs iterations

#generate data
y <- rnorm(n, mu.true, s2.true)
y.mean <- mean(y)

#starting point of the algorithm
mu <- 0
s2 <- 1

# Chains 
mu.chain <- c()
s2.chain <- c()

#### With constant prior on mean and std
for(i in 1: M){
  # Sampling m mu from full conditional
  mu <- rnorm(1, y.mean, sqrt(s2/n))
  # Sample s2 from its full conditional
  y.s <- sum((y - mu)^2)
  s2 <- 1/rgamma(1, n/2-1, y.s/2)
  
  # Store the values extracted
  mu.chain <- c(mu.chain, mu)
  s2.chain <- c(s2.chain, s2)
  if(i%%500 == 0){
    plot(mu.chain, s2.chain, type="b", col=4)
    Sys.sleep(0.1)
  }
}

mu.hat <- mean(mu.chain) # Estimate of mu
s2.hat <- mean(s2.chain) # esrtimate of s2
mu.hat
s2.hat

hist(mu.chain, 100, probability=TRUE, col=2) # plot of marginal posterior density of mu
abline(h=1, col=1, lwd=2) # Starting point was just constant

# To check if the output is correctly a sample from
# joint posterior distribution
# We check three things
# 1. Convergence to a stationary distribution
# 2. Convergence of the averages
# 3. IID sampling

# First coerce the output of chains in a mcmc object
# to be able to use the coda package
library(coda)
out <- mcmc(cbind(mu.chain, s2.chain))
plot(out)

# 1. Fist visualize trace plots
# and also a test, the Geweke test to assess it
# sort of two-sample t-test where you compare means in the first and
# last parts of the chains. The idea is that is stationarity
# then these two means should be same

gew <- coda::geweke.diag(out)
gew

# H0: Stationarity
gew$z # Realized values of test statistic(approx normal)
2*(1 - pnorm(abs(gew$z))) # p-values

#reject H0 if lower than a preficed signal level
# ---> we have reached stationarity

# this test is used as a preliminary tool to understand
# if you need more burn-in

geweke.plot(out) 
# test statistic as we increase number 
# of discarded iterations at the beginning to understand
# if further burn-in is required 

# 2. Convergence of means 
coda::cumuplot(out)
# Plot of the cumulative means
# with 2.5% and 97.5% quantiles


# 3. iid?
coda::autocorr.diag(out)
coda::autocorr.plot(out)

coda::batchSE(out) # To compute appropriate standard error for the
# MCMC estimators that takes into account
# the presence of correlations among samples extracted


#### Now change priors. Normal for mean and inverse-gamma for standard deviation

# With informative normal for mu and inverse gamma for std priors

# Fix the hyperpars for priors
mu0 <- 0; s20 <- 0.1;
v <- s0 <- 1;

  
#starting point of the algorithm
mu <- 0
s2 <- 1

# Chains 
mu.chain <- c()
s2.chain <- c()

#### With informative(non-constant) prior on mean and std
for(i in 1: M){
  # Sampling m mu from full conditional
  mu.var <- 1/(n/s2 + 1/s20)
  mu.mean <- mu.var * (y.mean/s2 + mu0/s20)
  mu <- rnorm(1, mu.mean, mu.var)
  
  # Sample s2 from its full conditional
  y.s <- sum((y - mu)^2)
  s2 <- 1/rgamma(1, n/2 + v/2, y.s/2 + s0/2)
  
  # Store the values extracted
  mu.chain <- c(mu.chain, mu)
  s2.chain <- c(s2.chain, s2)
  if(i%%500 == 0){
    plot(mu.chain, s2.chain, type="b", col=4)
    Sys.sleep(0.1)
  }
}

hist(mu.chain, 50, freq = FALSE,
     xlim=c(-2,2)) # marginal Posterior
plot(function(mu) dnorm(mu, mu0, sqrt(s20)),
      add = TRUE, col=1, lwd=1, from=-2, to=2)

# Plotting density of prior of mean with plot over histogram
# For doing the same thing for plotting prior on variance 
# i.e inverse gamma density use package invgamma
# and avoid the jacobian you should consider if your transform the gamma

library("invgamma")

hist(s2.chain, 1000, freq = FALSE,
     xlim=c(0,10)) # marginal Posterior
plot(function(s2) dinvgamma(s2, v/2, s0/2),
     add = TRUE, xlim=c(0,10))

out <- mcmc(cbind(mu.chain, s2.chain))
plot(out)

gew <- coda::geweke.diag(out)
gew

# H0: Stationarity
gew$z # Realized values of test statistic(approx normal)
2*(1 - pnorm(abs(gew$z))) # p-values

geweke.plot(out) 

coda::cumuplot(out)

coda::autocorr.diag(out)
coda::autocorr.plot(out)


############### Example2: Gibbs Sampler page 84 ##################

# fix the setting
n <- 20
M <- 10000

# Fix hyperparams
alpha <- a <- b <- 10


beta.true <- rninvgamma(n, a, b)
lambda.true <- rgamma(n, alpha, scale = beta.true)

x   <- rpois(n, lambda = lambda.true) # This is obevsable


# staring points
lam <- bet <- rep(1, n)
# chains
lambda.chain <- beta.chain <- c()

for (i in 1:M){
  # Sample lambda_i ,i=1,.....,n
  lam <- rgamma(n, alpha+x, scale=1/(1 + 1/bet))
  # Sample beta_i, i=1,......,n
  bet <- rinvgamma(n, alpha+a, lam + b)
  # Store values
  lambda.chain <- cbind(lambda.chain, lam)
  beta.chain <- cbind(beta.chain, bet)
}

lambda.hat <- rowMeans(lambda.chain)
beta.hat <- rowMeans(beta.chain)
beta.hat


