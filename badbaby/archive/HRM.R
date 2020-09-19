# This is code to carry out hierarchical regression modeling of infant mismatch negativity paradigm. 
# author = Kambiz Tavabi
# credits = Rechele Brooks
# copyright = Copyright 2018, Seattle, Washington
# license = MIT
# version = 0.1.0
# maintainer = Kambiz Tavabi
# email = ktavabi@uw.edu
# status = Development

# Print func https://rpubs.com/bbolker/4619
pr <- function(m) printCoefmat(coef(summary(m)),digits=3,signif.stars=T)

# TODO How do I assess random effects in random intercept (centering) and/or slope? I.e. how do I determine whether random intercepts and/or slope is necessary in the model.
# TODO Sort data by ID then age
# TODO Use NLMER for covariance structure e.g., Compound symmetry for (equal) covariance structure preferred for RM paradigm
# TODO Center at 12mos and using both linear & quadratic growth models
# TODO Center at 30mos using linear (quadratic?) 
# Centering point conveys the predictive logic i.e., predicting early vs. late

library(tidyverse)
# Read in data
dfs <- merge(read_csv("Ds-mmn-2mos_meg_df.csv"),
             read_csv("Ds-mmn-2mos_cdi_df.csv"),
             by = "subjId") %>% 
    select(c(subjId, stimulus, hemisphere, oddballCond, gender.x, age.x,
             headSize.x, birthWeight.x, hemisphere, auc, latencies, cdiAge, 
             m3l, vocab, ses.y, sesGroup.y, maternalHscore.y))

# factorize columns 
# (https://gist.github.com/ramhiser/93fe37be439c480dc26c4bed8aab03dd)
dfs <- dfs %>%
    mutate(hemisphere = as.character(hemisphere),
           char_column = sample(letters[1:5], nrow(dfs), replace = TRUE))
sum(sapply(dfs, is.character))
dfs <- dfs %>%
    mutate_if(sapply(dfs, is.character), as.factor)
sapply(dfs, class)

# center scale vars
dfs <- mutate(dfs, c_ses = scale(ses.y, scale = F))  
dfs <- mutate(dfs, c_maths = scale(maternalHscore.y, scale = F))
dfs <- mutate(dfs, c_logAuc = scale(log10(auc), scale = F))
dfs <- mutate(dfs, c_latencies = scale(latencies, scale = F))
dfs <- mutate(dfs, c_cdiAge = scale(cdiAge, center = 24))  # center CDI at 24
dfs <- mutate(dfs, c_vocab = scale(vocab, scale = F))  
dfs <- mutate(dfs, c_m3l = scale(m3l, scale = F))  

str(dfs) # structure of the data

print('participants per oddball condition:')
dfs %>% group_by(subjId, oddballCond) %>%
    summarise(n = n_distinct(subjId)) %>%
    ungroup() %>%
    group_by(oddballCond) %>%
    summarise(n = n())

## are all participants in three stimulus conditions?
dfs %>% group_by(subjId) %>%
    summarise(task = n_distinct(stimulus)) %>%
    as.data.frame() %>% 
    {.$task == 3} %>%
    all()

# Modeling
# https://cran.r-project.org/web/packages/afex/vignettes/afex_mixed_example.html
library("afex") # needed for mixed() and attaches lme4 automatically.
afex_options(emmeans_model = "multivariate")  # ANOVAs involving RM factors, follow-up tests based on the multivariate model are generally preferred to univariate follow-up tests.
set_sum_contrasts()  # orthogonal sum-to-zero contrasts

########################
# VOCAB
# Random intercept with fixed mean.
pr(vocab.null <- mixed(vocab ~ 1 + (1|subjId), 
                       dfs, method = "S", 
                       control = lmerControl(optCtrl = list(maxfun = 1e6)),
                       expand_re = TRUE))

# Unconditional growth model (one way Anova) w/ uncorrelated random intercept and slope.
pr(vocab.fm1 <- mixed(vocab ~ c_cdiAge +  (c_cdiAge||subjId), 
                      dfs, method = "S", 
                      control = lmerControl(optCtrl = list(maxfun = 1e6)),
                      expand_re = TRUE))

# Same as above but with correlated random intercept and slope.
pr(vocab.fm2 <- mixed(vocab ~ c_cdiAge + (c_cdiAge|subjId), 
                      dfs, method = "S", 
                      control = lmerControl(optCtrl = list(maxfun = 1e6)),
                      expand_re = TRUE))
pr(vocab.fm3 <- mixed(vocab ~ c_cdiAge * gender.x + (c_cdiAge|subjId), 
                      dfs, method = "S", 
                      control = lmerControl(optCtrl = list(maxfun = 1e6)),
                      expand_re = TRUE))
pr(vocab.fm4 <- mixed(vocab ~ c_cdiAge * c_maths * gender.x + (c_cdiAge|subjId), 
                      dfs, method = "S", 
                      control = lmerControl(optCtrl = list(maxfun = 1e6)),
                      expand_re = TRUE))
pr(vocab.fm5 <- mixed(vocab ~ c_cdiAge * c_ses * gender.x + (c_cdiAge|subjId), 
                      dfs, method = "S", 
                      control = lmerControl(optCtrl = list(maxfun = 1e6)),
                      expand_re = TRUE))
# Maximal model with correlated random intercept and slope.
# Fixed effects: Age * SES + Latency * Stimulus * Hemisphere
# Random effects: Age|ID + Stimulus * Hemisphere|ID
pr(vocab.fm6 <- mixed(vocab ~ c_cdiAge * c_ses +
                          c_latencies * oddballCond * hemisphere +
                          (c_cdiAge|subjId) + 
                          (oddballCond * hemisphere|subjId), 
                      dfs, method = "S", 
                      control = lmerControl(optCtrl = list(maxfun = 1e6)),
                      expand_re = TRUE))

anova(vocab.null.isa1, vocab.isa1, test="F")

########################
# M3L
# by-s random intercepts
pr(m3l.null.is <- mixed(m3l ~ 1 + (1|subjId), 
                        dfs, method = "S", 
                        control = lmerControl(optCtrl = list(maxfun = 1e6)),
                        expand_re = TRUE))
# by-s random intercepts and by-s random slopes for a plus their correlation
pr(m3l.null.isa <- mixed(m3l ~ 1 + (c_cdiAge|subjId), 
                         dfs, method = "S", 
                         control = lmerControl(optCtrl = list(maxfun = 1e6)),
                         expand_re = TRUE))
# by-s random intercepts and by-s random slopes without their correlation
pr(m3l.null.isa1 <- mixed(m3l ~ 1 + (c_cdiAge|subjId), 
                          dfs, method = "S", 
                          control = lmerControl(optCtrl = list(maxfun = 1e6)),
                          expand_re = TRUE))
# Unconditional growth model conveying Age is the best predictor of measure i.e. one way Anova
pr(m3l.isa <- mixed(m3l ~ c_cdiAge + (c_cdiAge|subjId), 
                    dfs, method = "S", 
                    control = lmerControl(optCtrl = list(maxfun = 1e6)),
                    expand_re = TRUE))
# Maximal model with by-s random intercepts and by-s random slopes with their correlation
pr(m3l.isa1 <- mixed(m3l ~ c_cdiAge + 
                         latencies * oddballCond * hemisphere +
                         logauc * oddballCond * hemisphere +
                         s_ses +
                         (c_cdiAge|subjId) + 
                         (oddballCond * hemisphere|subjId), 
                     dfs, method = "S", 
                     control = lmerControl(optCtrl = list(maxfun = 1e6)),
                     expand_re = TRUE))
anova(m3l.null.isa1, m3l.isa1, test="F")
