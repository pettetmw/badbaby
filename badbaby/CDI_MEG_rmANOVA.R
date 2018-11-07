# Title     : Repeated Measures ANOVA
# Objective : Model selection for MMN-Dataset
# Created by: ktavabi
# Created on: 11/5/18

###########################
# Repeated Measures ANOVA #
###########################
# When an experimental design takes measurements on the same experimental unit # over time, the
# analysis of the data must take into account the probability that measurements for a given
# experimental unit will be correlated in some way. One approach to deal with non-independent
# observations is to include an autocorrelation structure in the model using
# the nmle package, e.g. treating Subject_ID as a random variable to take into account
# native differences among subjects
# https://rcompanion.org/handbook/I_09.html

# Install packages if not already installed
if(!require(psych)){install.packages("psych")}
if(!require(nlme)){install.packages("nlme")}
if(!require(car)){install.packages("car")}
if(!require(multcompView)){install.packages("multcompView")}
if(!require(lsmeans)){install.packages("lsmeans")}
if(!require(ggplot2)){install.packages("ggplot2")}
if(!require(rcompanion)){install.packages("rcompanion")}

# Imports
library(psych)
library(nlme)
library(car)
library(rcompanion)
library(multcompView)
library(emmeans)
library(rcompanion)
library(ggplot2)

MMNdf <- read.csv(file = "/home/ktavabi/Projects/badbaby/badbaby/static/MMNdf_RM.csv",
header=TRUE, sep="\t")
MMNdf.mean <- aggregate(MMNdf$latencies,
by=list(MMNdf$Subject_ID, MMNdf$ses_label,
MMNdf$condition_label, MMNdf$hem_label),
FUN='mean')

# Order factors by the order in data frame, otherwise, R will alphabetize them
MMNdf$Ses = factor(MMNdf$SES,
levels = unique(MMNdf$SES))

# Check the data frame
headTail(MMNdf)
str(MMNdf)
summary(MMNdf)

######################
# Mixed effect model #
######################
# Fit a linear mixed-effects model in the formulation described in Laird and
# Ware (1982) but allowing for nested random effects. The within-group errors
# are allowed to be correlated and/or have unequal variances. Model is fit by
# maximizing the restricted log-likelihood (REML).
# Covariance structure corAR1 with form ~1|subjvar is used to indicate a
# temporal autocorrelation structure of order one, where time covariate is
# order of observations, and Subject_ID indicates the variable for
# experimental units.  Autocorrelation is modeled within levels of the
# subjvar, and not between them.
# NB Autocorrelation structures can be chosen by either of two methods. The
# first is to choose a structure based on theoretical expectations of how the
# data should be correlated. The second is to try various autocorrelation
# structures and compare the resulting models with a criterion like AIC,
# AICc, or BIC to choose the structure that best models the data.

fmla <- as.formula("latencies ~ ses_label + condition_label + hem_label +
                    ses_label * condition_label + ses_label * hem_label +
                    condition_label * hem_label + ses_label *
                    condition_label * hem_label")

# Mixed effect model
model.mixed = lme(fmla, random=~ 1 | Subject_ID,
correlation=corAR1(form=~ 1 | Subject_ID), data=MMNdf, method="REML")

summary(model.mixed)
Anova(model.mixed)

######################
# Fixed effect model #
######################
model.fixed = gls(fmla, correlation=corAR1(form=~1|Subject_ID),
data=MMNdf, method='REML')

summary(model.fixed)
Anova(model.fixed)

##############################
# Model comparison/selection #
##############################
# The random effects in the model can be tested by comparing the model to a
# model fitted with just the fixed effects and excluding the random effects.
# Because there are not random effects in this second model, the gls function
# in the nlme package is used to fit this model.

anova(model.mixed, model.fixed) # compares the reduction in the residual sum of squares
# NB the residual sum of squares (RSS), also known as the sum of squared
# residuals (SSR) or the sum of squared errors of prediction (SSE), is the sum
# of the squares of residuals (deviations predicted from actual empirical
# values of data). It is a measure of the discrepancy between the data and an
# estimation model. A small RSS indicates a tight fit of the model to the data.
# It is used as an optimality criterion in parameter selection and model
# selection.

###########################################################
# Pseudo R^2 Measures of Fit for fixed model of latencies #
###########################################################
# The nagelkerke function can be used to calculate a p-value and pseudo
# R-squared value for the model. One approach is to define the null model as
# one with no fixed effects except for an intercept, indicated with a 1 on
# the right side of the ~.  And to also include the random effects, in this
# case 1|Subject_ID.
# A signficant pseudo R-squared indicates this model better fits the outcome
# data than the Null model. While pseudo R-squareds cannot be interpreted
# independently or compared across datasets, they are valid and useful in
# evaluating multiple models predicting the same outcome on the same dataset.
# In other words, a pseudo R-squared statistic without context has little
# meaning. A pseudo R-squared only has meaning when compared to another
# pseudo R-squared of the same type, on the same data, predicting the same
# outcome.  In this situation, the higher pseudo R-squared indicates which
# model better predicts the outcome.
# https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faq-what-are-pseudo-r-squareds/

model.null = lme(latencies ~ 1, random=~1|Subject_ID, data=MMNdf)
nagelkerke(model.fixed, model.null)

#####################
# Post-Hoc analysis #
#####################
marginal = emmeans(model.fixed, ~ ses_label : hem_label)
cld(marginal, alpha=0.05, Letters=letters, ### Use lower-case letters for .group
    adjust="tukey")     ###  Tukey-adjusted comparisons

Sum = groupwiseMean(latencies ~ ses_label + hem_label,
    data=MMNdf, conf=0.95, digits=3, traditional=FALSE, percentile=TRUE)

Sum

pd = position_dodge(.2)
ggplot(Sum, aes(x=hem_label, y=Mean, color=ses_label)) +
    geom_errorbar(aes(ymin=Percentile.lower,
        ymax=Percentile.upper),
        width=.2, size=0.7, position=pd) +
    geom_point(shape=15, size=4, position=pd) +
    theme_bw() +
    theme(axis.title=element_text(face="bold")) +
    ylab("Peak Latency")
