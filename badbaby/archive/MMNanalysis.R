# Infant MEG-mismatch negativity paradigm inferential analysis script
# author=Kambiz Tavabi
# credits = Rechele Brooks, Daniel Hippe
# copyright=Copyright 2018, Seattle, Washington
# license=MIT
# version=0.1.0
# maintainer=Kambiz Tavabi
# email=ktavabi@uw.edu
# status=Development

library(tidyverse)
##############
# Read in data
dfs <- merge(read_csv("Ds-mmn-2mos_meg_df.csv"),
             read_csv("Ds-mmn-2mos_cdi_df.csv"),
             by="subjId") %>% 
    select(c(subjId, stimulus, hemisphere, oddballCond, gender.x, age.x,
             headSize.x, birthWeight.x, hemisphere, auc, latencies, cdiAge, 
             m3l, vocab, ses.y, sesGroup.y, maternalHscore.y))

str(dfs) # structure of the data
# participants per oddball condition
dfs %>% group_by(subjId, oddballCond) %>%
    summarise(n=n_distinct(subjId)) %>%
    ungroup() %>%
    group_by(oddballCond) %>%
    summarise(n=n())


#####
# box-whisker: latency
bx1 <- ggplot(dfs, aes(x=hemisphere, y=latencies, fill=oddballCond)) 
bx1 + geom_boxplot(notch=TRUE, outlier.colour="#E60704", outlier.shape=8,
                   outlier.size=4, alpha=.8,
                   position=position_dodge(width=.9)) + 
    geom_point(shape=20, alpha=.2,
               position=position_jitterdodge(dodge.width=0.9)) +
    stat_summary(geom="point", fun.y=mean, shape=18, 
                 size=3, position=position_dodge(width=.9),
                 show.legend=FALSE) +
    labs(x="Hemisphere",
         y="Latency (in seconds)",
         color="Stimulus") +
    theme_classic() + 
    scale_fill_manual(values=c("#0EF1F1", "#EC7D7B")) +
    scale_x_discrete(labels=c("left", "right")) +
    theme(axis.title.y=element_text(face="bold", angle=90),
          axis.title.x=element_text(face="bold"))


#####
# RM ANOVA: latencies
library("emmeans")  # emmeans must now be loaded explicitly for follow-up tests.
library("multcomp") # for advanced control for multiple testing/Type 1 errors.
library(afex) # needed for mixed() and attaches lme4 automatically.
afex_options(emmeans_model="multivariate")  # ANOVAs involving RM factors, follow-up tests based on the multivariate model are generally preferred to univariate follow-up tests.
set_sum_contrasts()  # orthogonal sum-to-zero contrasts

aov1 <- dfs %>% 
    aov_4(latencies ~ hemisphere + oddballCond + ((hemisphere + oddballCond)|subjId),
          anova_table=list(es="pes", correction="GG"),
            data=.)
aov1
m1 <- emmeans(aov1, ~ hemisphere + oddballCond)
summary(as.glht(pairs(m1)), test=adjusted("free"))


#####
# box-whisker: amplitude
bx2 <- ggplot(dfs, aes(x=hemisphere, y=auc, fill=oddballCond)) 
bx2 + geom_boxplot(notch=TRUE, outlier.colour="#E60704", outlier.shape=8,
                   outlier.size=4, alpha=.8,
                   position=position_dodge(width=.9)) + 
    geom_point(shape=20, alpha=.2,
               position=position_jitterdodge(dodge.width=0.9)) +
    stat_summary(geom="point", fun.y=mean, shape=18, 
                 size=3, position=position_dodge(width=.9),
                 show.legend=FALSE) +
    labs(x="Hemisphere",
         y="Log Amplitude (AU)",
         color="Stimulus") +
    theme_classic() + 
    scale_fill_manual(values=c("#0EF1F1", "#EC7D7B")) +
    scale_x_discrete(labels=c("left", "right")) +
    scale_y_continuous(trans='log10')
theme(axis.title.y=element_text(face="bold", angle=90),
      axis.title.x=element_text(face="bold"))


#####
# RM ANOVA: amplitude
aov2 <- dfs %>% 
    aov_4(latencies ~ hemisphere + oddballCond + ((hemisphere + oddballCond)|subjId),
          anova_table=list(es="pes", correction="GG"),
          data=.)
aov2
m2 <- emmeans(aov2, ~ hemisphere + oddballCond)
summary(as.glht(pairs(m1)), test=adjusted("free"))



#####
# CDI 
# densities
library(ggthemes)
library(lattice) # for plots
library(latticeExtra) # for combining lattice plots, etc.
lattice.options(default.theme = standard.theme(color = FALSE)) # black and white
lattice.options(default.args = list(as.table = TRUE)) # better ordering
# Change some of the default lattice settings
my.settings <- canonical.theme(color = FALSE)
my.settings[['strip.background']]$col <- "black"
my.settings[['strip.border']]$col<- "black"
cdi.long <- gather(dfs, "type", "response", m3l, vocab)
histogram(~response|type, cdi.long, breaks = "Scott", type = "density",
          scale = list(x = list(relation = "free"), y = list(relation = "free")),
          xlab = "M3L",
          panel = function(x, ...) {
              panel.histogram(x, ...)
              xn <- seq(min(x), max(x), length.out = 100)
              yn <- dnorm(xn, mean(x), sd(x))
              panel.lines(xn, yn, col = "red")
          })


#####
# spaghetti plots
y <- filter(cdi.long, response > 0)$response
lambda2 <- boxcoxfit(y, lambda=0.5)
lambda2[["lambda"]]
cdi.long <- mutate(cdi.long, logResp = log10(response + lambda2[["lambda"]]))
sp <- ggplot(data = cdi.long, aes(x = cdiAge, y = logResp, group = subjId, color = sesGroup.y))
sp + geom_line(size = .8, alpha = .5) +
    stat_smooth(aes(group = 1), method = "lm", se = TRUE, 
                size = 1, show.legend = FALSE) +
    stat_summary(aes(group = 1), geom = "point", fun.y = mean, shape = 18, 
                 size = 3, show.legend = FALSE) +
    facet_wrap(~ type, scales="free") +
    labs(title = "Language outcome", 
         subtitle = "Regression line ± error",
         x = "Age (in months)",
         y = "Log CDI",
         color = "SES") +
    theme_classic() + 
    scale_fill_manual(values = c("#0EF1F1", "#EC7D7B")) +
    theme(axis.title.y = element_text(face = "bold", angle=90),
          axis.title.x = element_text(face = "bold"))

#####
# box-whisker: CDI
bx3 <- ggplot(cdi.long, aes(x = cdiAge, y = logResp, fill = factor(cdiAge))) 
bx3 +  geom_boxplot(notch = TRUE, outlier.colour = "#E60704", outlier.shape = 8,
                   outlier.size = 4, alpha = .6) + 
    stat_summary(geom = "point", fun.y = mean, shape = 18, 
                 size = 3, show.legend = FALSE) +
    geom_jitter(shape = 20, alpha = .2, position = position_jitter(0.2)) +
    facet_wrap(~ type, scales="free") +
    labs(title = "Language outcome", 
         x = "Age (in months)",
         y = "Vocabulary size",
         fill = "Age") +
    theme_classic() + 
    theme(axis.title.y = element_text(face = "bold", angle = 90),
          axis.title.x = element_text(face = "bold"))
