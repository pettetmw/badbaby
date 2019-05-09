# Title     : Repeated Measures ANOVA
# Objective : Model selection for MMN-Dataset - 2 mos CDI responses
# Created by: ktavabi
# Created on: 11/5/18

table_glht <- function(x) {
    pq <- summary(x)$test
    mtests <- cbind(pq$coefficients, pq$sigma, pq$tstat, pq$pvalues)
    error <- attr(pq$pvalues, "error")
    pname <- switch(x$alternativ, less= paste("Pr(<", ifelse(x$dfs==0, "z", "t"), ")", sep= ""), 
                    greater= paste("Pr(>", ifelse(x$dfs== 0, "z", "t"), ")", sep= ""), two.sided= paste("Pr(>|",ifelse(x$dfs== 0, "z", "t"), "|)", sep= ""))
    colnames(mtests) <- c("Estimate", "Std. Error", ifelse(x$dfs==0, "z value", "t value"), pname)
    return(mtests)
}

suppressMessages(library(tidyverse))
dfs <- merge(read_csv("Ds-mmn-2mos_meg_df.csv"),
             read_csv("Ds-mmn-2mos_cdi_df.csv"),
             by = "subjId") %>% 
    select(c(subjId, stimulus, hemisphere, oddballCond, gender.x,
             hemisphere, auc, latencies, cdiAge, m3l, vocab, ses.y, sesGroup.y, maternalHscore.y))
dfs$logauc = log(dfs$auc)

# factor columns (https://gist.github.com/ramhiser/93fe37be439c480dc26c4bed8aab03dd)
dfs <- dfs %>%
    mutate(
        hemisphere = as.character(hemisphere),
        char_column = sample(letters[1:5], nrow(dfs), replace = TRUE))
sum(sapply(dfs, is.character)) # 2
dfs <- dfs %>%
    mutate_if(sapply(dfs, is.character), as.factor)
sapply(dfs, class)

# summarize data
(meg.summary <- group_by(dfs, sesGroup.y, oddballCond) %>%
        summarise_at(c("latencies", "auc"),
                     list(
                         mean = mean,
                         sd = sd,
                         min = min,
                         max = max,
                         iqr = IQR
                     ),
                     na.rm = TRUE))
(cdi.summary <- group_by(dfs, cdiAge, sesGroup.y) %>%
        summarise_at(c("vocab", "m3l"), 
                     list(mean=mean, 
                          sd=sd, 
                          min=min, 
                          max=max, 
                          iqr=IQR), 
                     na.rm=TRUE))




# CDI RM 2-way ANOVA (VOCAB, M3L) by SES over Gender. See
# https://egret.psychol.cam.ac.uk/statistics/R/anova.html#overview) and 
# http://coltekin.net/cagri/R/r-exercisesse5.html
suppressMessages(library(ez)) # for any of the ez* functions
dfs$cdiAge <- as.factor(dfs$cdiAge)
ezPrecis(dfs)
ezDesign(dfs,
         x=sesGroup.y,
         y=m3l,
         col=cdiAge)

(stats <- ezStats(data=dfs,
                  dv=.(m3l),
                  wid=.(subjId),
                  within=.(cdiAge),
                  between=.(sesGroup.y))) 

ezPlot(dfs, dv=.(m3l), 
       wid=.(subjId),
       within=.(cdiAge),
       between=.(sesGroup.y),
       x=.(cdiAge),
       x_lab="Age (months)",
       y_lab="M3L (M\u00B1SD)",
       split=.(sesGroup.y), 
       split_lab=c("SES"),
       do_lines=T, 
       do_bars=T,
       levels=list(
           cdiAge=list(
               new_names=c('18', '21','24','27','30')
           ),
           sesGroup.y=list(
               new_names=c('low', 'high'))),
       print_code=F
)

(anova <- ezANOVA(dfs, dv=.(m3l), 
                  wid=.(subjId), 
                  within=.(cdiAge),
                  within_full = .(cdiAge),
                  between=.(sesGroup.y, gender.x),
                  detailed=T, type="II", return_aov=T))

# Mixed models
suppressMessages(library(car))
suppressMessages(library(lme4))
# lmer
Anova(fm1 <- lmer(vocab~sesGroup.y * gender.x * cdiAge + (1|subjId), data=dfs))
# nlme
library(nlme)
Anova(fm2 <- lme(vocab~sesGroup.y * gender.x * cdiAge, 
                 random=~1|subjId/(cdiAge), data=dfs))

# Post-hoc comparisons 
# Age
library(emmeans)
marginal= emmeans(fm1, ~ cdiAge)
(vocab.emm <- cld(marginal,
                  alpha=0.05,
                  Letters=letters,  ### Use lower-case letters for .group
                  adjust="bonferroni"))  ###  bonferroni-adjusted comparisons
vocab.emm$.group=gsub(" ", "", vocab.emm$.group)   
library(rcompanion)
Sum=groupwiseMean(vocab ~ cdiAge,
                  data=dfs, conf=0.95, digits=3, boot=TRUE, bca=TRUE,
                  traditional=FALSE,
                  percentile=TRUE)
#EMM plot
library(ggplot2)
pd <-  position_dodge(0.4)
ggplot(vocab.emm,
       aes(
           x = cdiAge,
           y = lsmean,
           label = .group)) +
    geom_point(shape = 15,
               size = 4,
               position = pd) +
    geom_errorbar(
        aes(ymin = lower.CL, ymax = upper.CL),
        width = 0.2,
        size = 0.7,
        position = pd) +
    theme_bw() +
    theme(
        axis.title   = element_text(face = "bold"),
        axis.text    = element_text(face = "bold"),
        plot.caption = element_text(hjust = 0)) +
    ylab("Least square mean\n Vocabulary size") +
    xlab("Age (mo.)") +
    geom_text(nudge_x = c(rep_len(c(0.1, -0.1), length(vocab.emm$.group))),
              nudge_y = c(rep_len(3.5, length(vocab.emm$.group))),
              color = "grey") +
    scale_color_manual(values = mycols) +
    ggtitle ("Vocabulary size for CDI age") +
    labs(caption  = paste0("\nVocabulary size across ",
                           "five CDI measurements. Boxes indicate \n",
                           "the LS mean. ",
                           "Error bars indicate the 95% confidence ",
                           "interval ",
                           "of the LS \n",
                           "mean. Means sharing a letter are ",
                           "not significantly different \n",
                           "(Bonferroni-adjusted comparisons)."),
         hjust=0.5)


library(multcomp)
summary(gg <- glht(fm1, linfct=mcp(cdiAge= "Tukey"), test= adjusted(type= "bonferroni")))
plot(gg)
# Age:gender
suppressMessages(library(lsmeans))
lsmeans(fm1, pairwise ~ gender.x | cdiAge,
        adjust="tukey")
# Age:SES group 
lsmeans(fm1, pairwise ~ sesGroup.y | cdiAge,
        adjust="bonferroni")
# Interaction plot 
marginal <- lsmeans(fm1, ~ sesGroup.y:cdiAge)
(m3l.emm <- cld(marginal,
                alpha=0.05,
                Letters=letters,
                adjust="bonferroni"))
m3l.emm$.group=gsub(" ", "", m3l.emm$.group)
pd <-  position_dodge(0.4)
mycols <- c("#2E9FDF", "#FC4E07")
ggplot(m3l.emm,
       aes(
           x = cdiAge,
           y = lsmean,
           color = sesGroup.y,
           label = .group)) +
    geom_point(shape = 15,
               size = 4,
               position = pd) +
    geom_errorbar(
        aes(ymin = lower.CL, ymax = upper.CL),
        width = 0.2,
        size = 0.7,
        position = pd) +
    theme_bw() +
    theme(
        axis.title   = element_text(face = "bold"),
        axis.text    = element_text(face = "bold"),
        plot.caption = element_text(hjust = 0)) +
    ylab("Least square mean\nM3L") +
    geom_text(nudge_x = c(rep_len(c(0.1, -0.1), length(m3l.emm$.group))),
              nudge_y = c(rep_len(3.5, length(m3l.emm$.group))),
              color = "grey") +
    scale_color_manual(values = mycols) +
    ggtitle ("M3L for SES group and CDI age") +
    labs(caption  = paste0("\nM3L for two SES groups across ", 
                           "five CDI measurements. Boxes indicate \n",
                           "the LS mean. ", 
                           "Error bars indicate the 95% confidence ",
                           "interval ",
                           "of the LS \n",
                           "mean. Means sharing a letter are ",
                           "not significantly different \n",
                           "(Bonferroni-adjusted comparisons)."),
         hjust=0.5)
    
