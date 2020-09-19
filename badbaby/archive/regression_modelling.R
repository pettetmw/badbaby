library(dplyr)
open <- read.csv("mmn-55_cdi_RMds.csv")
df <- dplyr::filter(open, ag == 2)
str(df)

m0 <- glm(vocab ~ ses, data = df)
summary(m0)
plot(df$ses, df$vocab)
lines(df$ses, predict(m0))
par(mfrow = c(2, 2))
plot(m0)

m1 <- glm(vocab ~ ses * cdiAge, data = df)
summary(m1)
plot(m1)

m2 <- glm(vocab ~ ses * cdiAge + maternalEdu, data = df)
summary(m2)
plot(m2)


library(ggplot2)
ggplot(df,
       aes(
         x = ses,
         y = vocab,
         shape = factor(cdiAge),
         color = maternalEdu
       )) +
  geom_point() +
  stat_smooth(method = "lm",
              show.legend = T,
              se = F)

df2 <- mutate(df, c_ses = scale(ses, scale = F))
df2 <- mutate(df2, c_auc = scale(auc, scale = F))

library(lme4)
library(arm)
## Varying-intercept model
M0 <- lmer (vocab ~ 1 + (1 | subjId), data = df2)
display (M0)
## correlated varying- intercept & slope model
M1 <- lmer (vocab ~ ses * cdiAge + maternalEdu +
              (1 + cdiAge | subjId), data = df2)  ## fails to converge
display (M1)
summary(M1)
## Uncorrelated varying- intercept & slope model
M2 <- lmer (vocab ~ ses * cdiAge + maternalEdu +
              (1 | subjId) + (cdiAge - 1 | subjId),
            data = df2)
display (M2)
summary(M2)



library("afex") # needed for mixed() and attaches lme4 automatically.
afex_options(emmeans_model = "multivariate")  # ANOVAs involving RM factors, follow-up tests based on the multivariate model are generally preferred to univariate follow-up tests.
set_sum_contrasts()  # orthogonal sum-to-zero contrasts
#TODO center numericals
M <- mixed(
  vocab ~ ses * cdiAge + maternalEdu +
    (1 | subjId) + (cdiAge - 1 | subjId),
  data = df2,
  method = "S"
)
summary(M)
