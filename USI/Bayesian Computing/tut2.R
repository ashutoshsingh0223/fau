raw.data <- read.csv('~/Desktop/ashutosh/USI/Bayesian Computing/stud_visualizzazioni.csv', sep=";")
 
dim(raw.data)
head(raw.data)


xyt <- raw.data$ORE[-c(1:6)]# x(to be adjusted in bayesian model)
yyt <- raw.data$YT[-c(1:6)] # y - dependent variable
yyt


diffxyt <- diff(xyt) # xyt contains cumulative scores
# to get closer to IID take diff from succeeding element
diffyyt <- abs(diff(yyt))


plot(diffxyt, diffyyt)
res.lm <- lm(diffyyt~diffxyt) # frequentist linear model
res.lm



# to get closer to Guassian data( careful: We have countings)
# we can apply Box-Cox transformation
# Log: Special case of Box-Cox

library(caret)

box <- BoxCoxTrans(diffyyt)
y.tr <- predict(box, diffyyt)

plot(diffxyt, y.tr)

y.tr 
diffyyt


D <- data.frame(diffxyt, y.tr)
lambda <- box$lambda
lambda

res.lm <- lm(y.tr~diffxyt)
res.lm


shapiro.test(diffyyt) # Strongly rejecting h0(data is guassian) very low p-value
shapiro.test(y.tr) 


prediction.freq <- predict(res.lm,
                           newdata = data.frame(diffxyt=3))
#predict y.tr when 3 hours have elapsed
# still in transformed space
# inverse the transform
prediction.freq <- (prediction.freq * lambda + 1)^(1/lambda)
prediction.freq
library(rstanarm)
model <- stan_glm(y.tr~diffxyt,chain=4, iter=2000,
                  warmup=250, data=D)
prior_summary(model)
# To use flat priors
model <- stan_glm(y.tr~diffxyt,chain=4, iter=2000,
                  warmup=250, data=D, prior = NULL,
                  prior_intercept = NULL,
                  prior_aux = NULL)

prior_summary(model)


# posterior distribution
posteriors <- insight::get_parameters(model)
# 7000 = 4 *(2000 - 250)
posteriors
mean(posteriors$diffxyt) #posterior mean on beta(slope)
bayestestR::map_estimate(posteriors$diffxyt)


mean(posteriors$`(Intercept)`) #posterior mean on alpha
bayestestR::map_estimate(posteriors$`(Intercept)`)

# See the posterior
library(ggplot2) 

ggplot(posteriors, aes(x = diffxyt)) + geom_density(fill="orange") # Posterior density of beta

ggplot(posteriors, aes(x = `(Intercept)`)) + 
  geom_density(fill="orange") +
  geom_vline(xintercept = mean(posteriors$`(Intercept)`), size=1) +# Posterior density of alpha
  geom_vline(xintercept = bayestestR::map_estimate(posteriors$`(Intercept)`), size=1, col=3)# Posterior density of alpha


# highest density regions(euivalent of confidence interval in bayesian setting)
library(bayestestR)
Hdi <- hdi(posteriors$diffxyt, ci=0.8) # with posterior probability 95% beta is in this level
Hdi
Hdi$CI_low
Hdi$CI_high

ggplot(posteriors, aes(x = diffxyt)) +
  geom_density(fill="orange") +
  geom_vline(xintercept = Hdi$CI_low, size=1) +
  geom_vline(xintercept = Hdi$CI_high, size=1, col=2)
  
range(posteriors$diffxyt)


prediction.bayesian <- predict(model, newdata=data.frame(diffxyt=3))
prediction.bayesian # transformed scale

prediction.bayesian <- (prediction.bayesian * lambda + 1)^(1/lambda)

prediction.bayesian




########################################################
##################### Poisson model ####################
########################################################

# Use a GLM where likelihood is poisson
glm.freq <- glm(diffyyt~diffxyt, family=poisson)

# null hypotheses are that coeff of x is zero( y is not related to x) and
# intercept is zero( both are being rejected with very low p-vales)
# run below line to understand more
summary(glm.freq) # lower AIC is better

# beta: portion of expected counts due to x(elapsed time, time in poisson distribution, x of the data)
# alpha: remaining portion(prob due to missing co-variates, intercept)

prediction.glm.freq <- predict(glm.freq,
                               newdata = data.frame(diffxyt=3),
                               type='response')
prediction.glm.freq


# run bayesian poisson model
model.glm <- stan_glm(diffyyt~diffxyt,chain=4, iter=2000,
                  warmup=250, data=D, family="poisson")
prior_summary(model.glm)
#change to flat or to uninformative priors
# to see more on how to fix priors

help(priors,package='rstanarm')




# extract posteriors
posteriors.glm <- insight::get_parameters(model.glm)
mean(posteriors.glm$diffxyt) # posterior mean of beta

# try yourself all the steps done in bayesian lm

prediction.gml.bayesian <-predict(model,
                                  newdata = data.frame(diffxyt=3),
                                  type="response")
prediction.gml.bayesian
