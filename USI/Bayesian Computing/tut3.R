#### EXAMPLE-1 on Monte Carlo ####

Nsim <- 10^4
h <- function(x)(cos(50*x) + sin(20*x))^2 # function to compute integral of(integrand)

plot(h, lwd=2)

# Numerical deterministic solution
integrate(h, 0, 1) 


# MC estimator = 1/N sum(h(x_i)) ; x_i ~ Unif[0, 1]
x <- runif(Nsim) # extractions from unif
y <- h(x)
mean(y)

# Estimated integral
estint <- cumsum(y)/(1:Nsim)
plot(estint, type="l", lwd=2)

# Cumulative standard error of the estimator
esterr <- sapply(1: Nsim, function(i)sd(y[1:i])/sqrt(i))
esterr
lines(estint+2*esterr, col=2)
lines(estint-2*esterr, col=2)


### EXAMPLE-2 (see notes) ###
1 - pchisq(3/4*6, 6) #exact probability

# deriving the probability bt MC
set.seed(1234)
Mu <- 3 # Population mean
Sigma <- 2 #Population standard deviation
n <- 7 # Sample size

# Simulate Nsim scenarios for the sample
C <- 0 # Counter for S^2 > 3(sum of indicator values for I(s^2>3))


for (i in 1: Nsim){
  # Generate the sample
  x <- rnorm(n, Mu, Sigma)
  # Compute sample variance
  s2 <- var(x)
  # Check if condition is verified
  if (s2>3) C <- C+1
  
}

# Compute the mean of indicators
C / Nsim



#### EXAMPLE-3(Rejection Sampling) ####

alpha <- 2.7
beta <- 6.3

Nsim <- 100000
M <- optimize(function(x)dbeta(x, alpha, beta),
              interval = c(0, 1),
              maximum= TRUE)$obj
M

# Step1: Generate from the candidate distribution(called g in notes)
y <- runif(Nsim)

#Step2: generate from uniform
u <- runif(Nsim)

# Step3: discard those where u < target/(M*candidate)
x <- y[u<dbeta(y,alpha , beta)/M]
length(x)
length(x)/Nsim # Proportion of kept proposals

1/M
hist(x, 50, freq=FALSE) # Histogram of accepted values
plot(function(x)dbeta(x, alpha, beta), add=TRUE, col=2, lwd=2)


# Now try another closer candidate to avoid discarding
# so many proposals: chose g as beta(2, 6)

M <- optimize(function(x)dbeta(x, alpha, beta)/dbeta(x, 2, 6),
              interval=c(0,1),
              maximum = TRUE)$obj
M
1/M # probability of acceptance

# Step1 Generate from the candidate
y <- rbeta(Nsim, 2, 6)

# Step 2 sample from uniform
u <- runif(Nsim)

# Step3: discard those where u < target/(M*candidate)
x <- y[u < dbeta(y, alpha , beta)/(dbeta(y, 2, 6) * M)]
length(x)
length(x)/Nsim


hist(x, 50, freq=FALSE) # Histogram of accepted values
plot(function(x)dbeta(x, alpha, beta), add=TRUE, col=2, lwd=1)

par("mar")
par(mar=c(3,3,3,3))


#### EXAMPLE-4(Importance Sampling) ####
set.seed(12345)

Nsim <- 10^4
lambda <- 1 # first hyperparam
x0 <- y0 <- 0.5 # sec and third hyperparam
n <- 10 # sample size
alpha <- 1
beta <- 1
x <- rbeta(n, alpha, beta) # generate sample

# create a function for log posterior

logpost <- function(a, b){
  (lambda + n)*(lgamma(a + b) - lgamma(a) - lgamma(b)) +
    a*(log(x0) + sum(log(x))) + 
    b*(log(y0) + sum(log(1-x)))
}


a.grid <- seq(1, 5, length=200) # values for a for plot
b.grid <- a.grid
# i, j element of z.grid is logpost(a.grid[i], b.grid[j])
z.grid <- outer(a.grid, b.grid, logpost)

image(a.grid, b.grid, z.grid)

# chose t_3 scale distribution as canditate or importance distribution
s <- 1*diag(2) # cov matrix since two parameters are to be estimated in posterior i.e. a and b
# Play with variance to increase or decrease the supoort of g(importance distribution)

# Sample from t distribution, sigma=s, degrees of freedom=3, centre=2,2, since from `image(a.grid, b.grid, z.grid)`
# we see that our target distribution is roughly centred around 2,2
y <- mvtnorm::rmvt(Nsim, sigma=s, df=3,delta=c(2,2))
# Eliminate negative values, since a and b are positive, negative samples from t-dis will have
# zero posterior mass
miny <- apply(y, 1, min)
y <- y[miny > 0,]
length(y)
#points(y)

 # Compute importance sampling posterior mean

denom <- mvtnorm::dmvt(y, sigma=s, df=3, delta=c(2,2)) # in log

numera <- sapply(1:nrow(y), function(i)logpost(y[i,1], y[i, 2]))

# estimated posterior mean of a
mean(y[, 1]*exp(numera - denom)/mean(exp(numera - denom)))

# estimated posterior mean of a
mean(y[, 2]*exp(numera - denom)/mean(exp(numera - denom)))
