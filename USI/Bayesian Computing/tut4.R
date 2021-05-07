############### Example1: Variance Reduction ##################
N <- 10^4 # Number of extractions
h <- function(u)log(1 + log(1 - u) ^ 2)
u <- runif(N)

# MC solution
mean(h(u)) # Estimated expectation(integral value)
sd(h(u)) # related standard error

# Antithetic solution
mean((h(u) + h(1-u))/2)
sd((h(u) + h(1-u))/2) # related standard now lower

# Control Variates
b <- -cov(h(u), u)/var(u)
z <- h(u) + b*(u - 1/2)
mean(z)
sd(z) # related standard now lower



 ############### Example2: Markov Chains in R ##################
### Example on UBER drivers' movements ###

Driverzone <- c("N", "S", "W")
ZoneTransition <- matrix(c(0.3, 0.3, 0.4,
                           0.4, 0.4, 0.2,
                           0.5, 0.2, 0.3),
                         byrow=TRUE,
                         nrow = 3,
                         dimnames = list(Driverzone, Driverzone))

ZoneTransition
library(markovchain)
McZone <- new('markovchain', states=Driverzone,
              byrow=TRUE,
              transitionMatrix=ZoneTransition,
              name='Driver Movement')


McZone
summary(McZone) # Irreducible, All classes are recurrent
period(McZone) # Period = 1 implying chain is aperiodic

# Since MC is irreducible an aperiodic a unique stationary limiting 
# distribution

# Use package diagram for the plots
library(diagram)
plotmat(t(ZoneTransition), pos=c(1,2), box.size=0.05)

McZone^28
McZone^29

plot(sapply(1: 30,function(i)(McZone^i)[1, 1]), type="l",
     ylim=c(0.2, 0.4)) # From N to N
lines(sapply(1: 30,function(i)(McZone^i)[1, 2]), type="l",
     ylim=c(0.2, 0.4)) # From N to S
lines(sapply(1: 30,function(i)(McZone^i)[1, 3]), type="l",
      ylim=c(0.2, 0.4)) # From N to W

steadyStates(McZone) # Stationary Distribution


### Another example with data loaded with markovchain ###
data() ## Datasets available in R
data(rain)

rain
myseq <- rain$rain
myseq

createSequenceMatrix(myseq)
# Test H0: Null hypothesis that markov property exists
# A chi square distributed test statistic under H0
verifyMarkovProperty(myseq) # Big p-value --> do not reject H0
 
# Fit markov chain (classical way, that is MLE)
myFit <- markovchainFit(data=myseq, method="mle")
myFit

# Also fit markov chain (bayes way)
# with markovchain we cannot change prior, it uses conjugate prior on conditional
# probabilities(i.e the parameters)
# product of Dirichlet Distributions

# product is over different starting points i.e rows of the transition matrix
# default hyperpara: alpha=1;

myFit <- markovchainFit(data=myseq, method="map")
P.hat <- myFit$expectedValue # Estimated transition matrix
P.hat <- matrix(t(P.hat[]), nrow = 3, byrow = TRUE)

statesNames <- c('No rain', 'Light rain', 'heavy rain')

newMC <- new('markovchain', transitionMatrix=P.hat,
             states=statesNames)

summary(newMC) # Irreducible, all classes are recurrent
period(newMC) # Aperiodic.
#Since irreducible and aperiodic a unique staionary limiting distribution exists

steadyStates(newMC)
plotmat(t(P.hat), pos=c(1,2), box.size=0.05)


############### Example3: Metropolis Hastings Intro ##################
# Packages and functions for MCMC
# MCMCpack for MH algorithm
# In particular MCMCmetrop1R allows the implementation of MH
# on a user defined log posterior distribution
# In this function you have to specify log posterior
# and the initial value for the chain(for the parameters)
# burn-in (num of iters to discard)
# mcmc (iters to run after the burn-in)
# thin (keep 1 out of thin extractions)
# V for the variance of proposal distribiution

# package coda for convergence diagnostics
# package rstan for Hamiltonian MCMC and for NUTS(No U-turn Sampler) algorithm

 









###############  ##################