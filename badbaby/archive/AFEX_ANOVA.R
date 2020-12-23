# Title     : AFEX Repeated Measures ANOVA
# Objective : Inferential analysis of MMN-Dataset at 2 mos
# Created by: ktavabi
# Created on: 4/5/19

table_glht <- function(x) {
    pq <- summary(x)$test
    mtests <- cbind(pq$coefficients, pq$sigma, pq$tstat, pq$pvalues)
    error <- attr(pq$pvalues, "error")
    pname <- switch(x$alternativ, less= paste("Pr(<", ifelse(x$dfs==0, "z", "t"), ")", sep= ""), 
                    greater= paste("Pr(>", ifelse(x$dfs== 0, "z", "t"), ")", sep= ""), two.sided= paste("Pr(>|",ifelse(x$dfs== 0, "z", "t"), "|)", sep= ""))
    colnames(mtests) <- c("Estimate", "Std. Error", ifelse(x$dfs==0, "z value", "t value"), pname)
    return(mtests)
}

cols <- c("#2E9FDF", "#FC4E07")
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
dfs$cdiAge <- as.factor(dfs$cdiAge)
describeBy(dfs, dfs$sesGroup.y)

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

suppressMessages(library(psych))
## psych package to call for descriptives
(descriptives <- describeBy(dfs, dfs$sesGroup.y))

## ANOVA
suppressMessages(library(afex))
(vocab.aov <- aov_ez("subjId", "vocab", c("sesGroup.y", "gender.x"), 
                     c("cdiAge"),
                     anova_table = list(es = "pes", correction = "GG"),
                     return = afex_options("return_aov"),
                     data=dfs))
(m3l.aov <- aov_ez("subjId", "m3l", c("sesGroup.y", "gender.x"), 
                     c("cdiAge"),
                     anova_table = list(es = "pes", correction = "GG"),
                     return = afex_options("return_aov"),
                     data=dfs))
(latencies.aov <- aov_ez("subjId", "latencies",
                         between = c("sesGroup.y", "gender.x"),
                         within = c("hemisphere"),
                         anova_table = list(es = "pes", correction = "GG"),
                         return = afex_options("return_aov"),
                         data=filter(dfs, oddballCond == "deviant")))
(auc.aov <- aov_ez("subjId", "auc",
                   transformation = c("log"),
                   between = c("sesGroup.y", "gender.x"),
                   within = c("hemisphere", "oddballCond"),
                   anova_table = list(es = "pes", correction = "GG"),
                   return = afex_options("return_aov"),
                   data=dfs))

## Post-hoc analysis
# Vocabulary
library(emmeans)
vocab.emms <- emmeans(vocab.aov, ~cdiAge)
vocab.pairs <- cld(vocab.emms,
                   alpha=0.05,
                   Letters=letters,  ### Use lower-case letters for .group
                   adjust="bonferroni")  ###  bonferroni-adjusted comparisons
vocab.pairs$.group=gsub(" ", "", vocab.pairs$.group, fixed = T)  
print(vocab.pairs)
# plot
library(ggplot2)
library(ggrepel)
pd <-  position_dodge(0.5)
ggplot(vocab.pairs, aes(x = cdiAge, y = emmean, label = .group)) +
    geom_point(shape = 21, size = 4, position = pd) +
    geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL),
        width = 0.2, size = 0.5, position = pd) +
    theme_bw() +
    theme(axis.title = element_text(face = "bold"),
          axis.text = element_text(face = "bold"),
          plot.caption = element_text(hjust = 0)) +
    ylab("Least square mean\n Vocabulary size") +
    scale_x_discrete("Age (Months)",
                     labels=c("X18"="18", "X21"="21", "X24"="24", "X27"="27", "X30"="30")) +
    geom_text(nudge_x = c(rep_len(c(0.15, -0.15), length(vocab.pairs$.group))),
              nudge_y = c(rep_len(3.5, length(vocab.pairs$.group))),
              color = "grey") +
    ggtitle ("Vocabulary size for CDI age") +
    labs(caption  = paste0("\nVocabulary size across ",
                           "five CDI measurements. \n",
                           "Circles indicate the LS mean. \n",
                           "Error bars indicate the 95% confidence ",
                           "interval ",
                           "of the LS mean.\n",
                           "Means sharing a letter are ",
                           "not significantly different \n",
                           "(Bonferroni-adjusted comparisons)."), 
         hjust=0.5)

# m3l
m3l.emms <- emmeans(m3l.aov, ~sesGroup.y:cdiAge)
m3l.pairs <- cld(m3l.emms,
                   alpha=0.05,
                   Letters=letters,  ### Use lower-case letters for .group
                   adjust="bonferroni")  ###  bonferroni-adjusted comparisons
m3l.pairs$.group <- str_replace_all(string=m3l.pairs$.group, pattern=" ", repl="")
print(m3l.pairs)
ggplot(m3l.pairs, aes(x = cdiAge, y = emmean, fill = sesGroup.y, label = .group),
       postion = pd) +
    geom_point(shape = 21, size = 4, position = pd) +
    geom_text_repel(aes(x = cdiAge, y = emmean, color = sesGroup.y), 
                    point.padding = unit(.3,'lines'),
                    box.padding = unit(.3, 'lines'),
                    position = pd,
                    segment.size = 0,
                    show.legend = F) +
    geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL, color = sesGroup.y),
                  width = 0.2, size = 0.5, position = pd, show.legend = F) +
    theme_bw() +
    theme(axis.title = element_text(face = "bold"),
          axis.text = element_text(face = "bold"),
          plot.caption = element_text(hjust = 0)) +
    ylab("Least square mean M3l") +
    scale_color_manual(values = cols) +
    scale_fill_manual(name="SES",values=cols) + 
    scale_x_discrete("Age (Months)",
                     labels=c("X18"="18", "X21"="21", "X24"="24", "X27"="27", "X30"="30")) +
    ggtitle ("M3l for CDI age over SES") +
    labs(caption  = paste0("\nMean utterance lengths across ",
                           "five CDI measurements. \n",
                           "Circles indicate the LS mean. \n",
                           "Error bars indicate the 95% confidence ",
                           "interval ",
                           "of the LS mean.\n",
                           "Means sharing a letter are ",
                           "not significantly different \n",
                           "(Bonferroni-adjusted comparisons)."), 
         hjust=0.5)

# latencies
lat.emms <- emmeans(latencies.aov, ~sesGroup.y:hemisphere)
lat.pairs <- cld(lat.emms,
                 alpha=0.05,
                 Letters=letters,  ### Use lower-case letters for .group
                 adjust="none")  ###  bonferroni-adjusted comparisons
lat.pairs$.group=gsub(" ", "", lat.pairs$.group, fixed = T)  
print(lat.pairs)
ggplot(lat.pairs, aes(x = hemisphere, y = emmean, fill = sesGroup.y, label = .group),
       postion = pd) +
    geom_point(shape = 21, size = 4, position = pd) +
    geom_text_repel(aes(x = hemisphere, y = emmean, color = sesGroup.y), 
                    position = pd,
                    point.padding=unit(1,'lines'),
                    direction = 'both',
                    segment.size = 0,
                    show.legend = F) +
    geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL, color = sesGroup.y),
                  width = 0.2, size = 0.5, position = pd, show.legend = F) +
    theme_bw() +
    theme(axis.title = element_text(face = "bold"),
          axis.text = element_text(face = "bold"),
          plot.caption = element_text(hjust = 0)) +
    ylab("Least square mean latency (s)") +
    scale_x_discrete("Hemisphere",
                     labels=c("lh"="Left", "rh"="Right")) +
    scale_color_manual(values = cols) +
    scale_fill_manual(name="SES",values=cols) + 
    ggtitle ("Deviant ERF latencies for hemisphere age over SES") +
    labs(caption  = paste0("\nPeak deviant evoked response latency ",
                           "across hemispheres for SES groups. \n",
                           "Circles indicate the LS mean. \n",
                           "Error bars indicate the 95% confidence ",
                           "interval ",
                           "of the LS mean.\n",
                           "Means sharing a letter are ",
                           "not significantly different \n",
                           "(Bonferroni-adjusted comparisons)."), 
         hjust=0.5)
