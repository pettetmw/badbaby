#SIM-MANN/BEZOS funded
#2-4mos infants with SES & CDI(18-30)
#double oddball paradigm: ba<-->Xa<-->wa VOT length split differentiating in AC as #indexed by logit classifier AUC reliability scoring. Scoring was done on epoched MEG #measurements of MMN activity to VOT contrast embedded in synthetic CVs. 
#
library(tidyverse)
library(janitor)
library(ggplot2)
library(hrbrthemes)
library(viridis)

# sample size
sample_size = dataset %>% filter(`Filter 3` == 1 &
                                     `Filter 4` == 1) %>%
    group_by(age_grouped) %>% summarize(num = n())

# violin plot of the AUC data grouped by age
base <- dataset %>%
    ggplot(aes(
        x = reorder(factor(age_grouped), age),
        y = AUC,
        fill = age_grouped
    )) + geom_violin(trim = F, show.legend = F)
base +
    stat_summary(
        fun.data = "median_hilow",
        geom = "pointrange",
        color = "black",
        show.legend = F
    ) +
    scale_fill_viridis(
        discrete = TRUE,
        breaks = c('two', 'four', 'six'),
        alpha = .8
    ) +
    theme_ipsum() +
    theme(legend.position = "top",
          plot.title = element_text(size = 15)) +
    labs(title = "Logistic classifier scores over age",
         x = "Age (mos)",
         y = "Score (AUC)")

library(lme4)
library(sjPlot)
library(sjmisc)

#simple contrast matrix 
#https://stats.idre.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/#SIMPLE
#XCKed with JAMOVI results for same response (zScored-AUC) with random covars: SES, HdSz, BWt, Age dataframe; and grouping factors (ordinal Age:2,4,6), (Gender:b,g),
#(Cond:A,B,C,D)
c<-contr.treatment(4)
my.coding<-matrix(rep(1/4, 12), ncol=3)
my.simple<-c-my.coding
my.simple

#Fixed intercept model ANOVA main effect of CONDITION{A,B,C,D} with RandEfx SID
m.null <- lmer(scale(AUC_zscore) ~ 1 + condition + (1 | sid),
               data = dataset,
               control = lmerControl(optCtrl = list(maxfun = 1e6,
                                                    optimizer = 'bobyqa')))

summary(m.null)
plot_model(m.null, type = 'std2', show.values = T, value.offset = .2, 
           axis.labels = "", title = "Classifier score (AUC) mlm coefficients",
           vline.color = 'red', show.intercept = T)
tab_model(m.null, show.stat = T, show.df = T)
anova(m.null)

#Full FIModel of Cond-by-Age interaction accounting for covars and gender 
m1 <- lmer(AUC_zscore ~ 1 + condition + scale(age) + scale(ses) + age_grouped + gender + headSize + birthWeight + condition:scale(age)+(1+scale(age) | sid ),
           data=dataset, contrasts = list(condition = my.simple,
                                          age_grouped = contr.sum(3),
                                          gender = contr.sum(2)),
           control = lmerControl(optCtrl = list(maxfun = 1e6,
                                                optimizer = 'bobyqa')))
summary(m1)
anova(m1)
tab_model(m.null, m1, show.aic = T, show.stat = T, show.df = T, show.fstat = T)
plot_model(m1, type = 'std2',
           show.values = T, value.offset = .2, 
           axis.labels = "", title = "Classifier score (AUC) mlm coefficients")
# AUC coeffecients from full model for Condition:Age interaction signifies (importantly?) condition3:deviant-mmn and condition4:
#Model comparison suggests improvement likely!!!
anova(m.null, m1, test="LRT")

