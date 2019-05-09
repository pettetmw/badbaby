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

# Mixed models
suppressMessages(library(car))
suppressMessages(library(lme4))
# lmer
lmer(vocab~sesGroup.y * gender.x * cdiAge + (1|subjId), data=dfs))
xtabs(~ subjId + cdiAge, dfs)
print(dotplot(reorder(sesGroup.y, vocab) ~ vocab, dfs,
              ylab = "SES", jitter.y = TRUE, pch = 21,
              xlab = "Vocabulary size",
              type = c("p", "a")))
summary(vocab.m0 <- lmer(vocab ~ 1 + (1|subjId), data=dfs))
summary(vocab.m1 <- lmer(vocab ~ sesGroup.y * gender.x +  (1|subjId), data=dfs))

# nlme
library(nlme)
summary(null.vocab.lme <- lme(vocab ~ 1, random=~1|subjId, data=dfs))
summary(baseline.vocab.lme <- lme(vocab ~ 1, random = ~1|subjId/cdiAge, data = dfs))  ## cdiAge nested in SubjId

