# Title     : MMN Dataset (2mo) Stats
# Created by: Kambiz Tavabi
# Created on: 04/09/19


# Read in data
suppressMessages(library(tidyverse))
dfs <- merge(read_csv("Ds-mmn-2mos_meg_df.csv"),
             read_csv("Ds-mmn-2mos_cdi_df.csv"),
             by = "subjId") %>% 
    select(c(subjId, stimulus, hemisphere, oddballCond, gender.x, age.x,
             headSize.x, birthWeight.x,
             hemisphere, auc, latencies, cdiAge, m3l, vocab, ses.y, sesGroup.y, 
             maternalHscore.y))
dfs$logauc = -log(dfs$auc)

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
suppressMessages(library(psych))
## psych package to call for descriptives
(descriptives <- describeBy(dfs, dfs$sesGroup.y)) 

# summarize data
(dem.summary <- group_by(dfs, sesGroup.y, gender.x) %>% 
        mutate(Count = n_distinct(subjId)) %>%
        group_by(sesGroup.y, gender.x, Count) %>%
        summarise_at(c("age.x", "headSize.x", "birthWeight.x", "ses.y"), 
                     list(
                         MEAN = mean,
                         SD = sd,
                         MIN = min,
                         MAX = max,
                         IQR = IQR
                     ),
                     na.rm = TRUE))
write_delim(dem.summary, delim = "\t", "dem_desc.tsv")

(meg.summary <- group_by(dfs, sesGroup.y) %>%
        mutate(Count = n_distinct(subjId)) %>% 
        group_by(sesGroup.y, oddballCond, hemisphere, Count) %>% 
        summarise_at(c("latencies", "auc"),
                     list(
                         MEAN = mean,
                         SD = sd,
                         MIN = min,
                         MAX = max,
                         IQR = IQR
                     ),
                     na.rm = TRUE))
write_delim(meg.summary, delim = "\t", "meg_desc.tsv")

(cdi.summary <- group_by(dfs %>% filter(vocab > 0 & m3l > 0), sesGroup.y, cdiAge) %>%
        mutate(Count = n_distinct(subjId)) %>%
        group_by(sesGroup.y, cdiAge, Count) %>%
        summarise_at(c("vocab", "m3l"), 
                     list(
                         MEAN = mean,
                         SD = sd,
                         MIN = min,
                         MAX = max,
                         IQR = IQR
                     ),
                     na.rm = TRUE))
write_delim(cdi.summary, delim = "\t", "cdi_desc.tsv")

# EDA
suppressMessages(library(ggplot2))
suppressMessages(library(patchwork))

cols <- c("#2E9FDF", "#FC4E07")

# plots of vocab > 21mos
vocab <- ggplot(dfs %>% filter(cdiAge  %in% c("21", "24", "27", "30")), 
                aes(x = cdiAge, y = vocab, color = sesGroup.y))
vocab.bdp <- vocab + geom_boxplot(width = 0.5, size = 0.4, 
                                  position = position_dodge(width = 0.9)) +
    scale_color_manual(values = cols, name = 'SES') +
    labs(title = "Vocabulary size",
         subtitle = "Grouped by age over SES",
         x = "Age (mo)", y = "Vocabulary size")

vocab.dens <- ggplot(dfs %>% filter(cdiAge  %in% c("21", "24", "27", "30")), 
                     aes(vocab)) + 
    geom_density(aes(fill = sesGroup.y), alpha=0.4) + 
    scale_fill_manual(values = cols, name = 'SES') +
    labs(title="Vocabulary size", 
         subtitle="Density plot grouped by SES over all ages",
         x="Vocabulary size",
         fill="SES")
(vocab.bdp | vocab.dens)

# plots of m3l > 21mos
m3l <- ggplot(dfs %>% filter(cdiAge  %in% c("21", "24", "27", "30")), 
              aes(x = cdiAge, y = m3l, color = sesGroup.y))
m3l.bdp <- m3l + geom_boxplot(width = 0.5, size = 0.4, 
                              position = position_dodge(width = 0.9)) +
    scale_color_manual(values = cols, name = 'SES') +
    labs(title = "M3L",
         subtitle = "Grouped by age over SES",
         x = "Age (mo)", y = "M3L")

m3l.dens <- ggplot(dfs %>% filter(cdiAge  %in% c("21", "24", "27", "30")), 
                   aes(m3l)) + 
    geom_density(aes(fill = sesGroup.y), alpha=0.4) + 
    scale_fill_manual(values = cols, name = 'SES') +
    labs(title="Mean utterance length", 
         subtitle="Density plot grouped by SES over all ages",
         x="M3L",
         fill="SES")
(m3l.bdp | m3l.dens)

# plots of peak latencies
lat <- ggplot(dfs, aes(x=oddballCond, y=latencies, fill=hemisphere))
lat.bpd <- lat + 
    geom_boxplot(alpha = 0.5, position = position_dodge(width = 0.9)) +
    facet_grid(. ~ sesGroup.y) +
    scale_color_manual(values = cols, name = 'Hemisphere') +
    labs(title = "Peak ERF latency",
         subtitle = "grouped by condition & hemisphere over SES",
         x = "Stimulus", y = "Time (s)")

lat.dens <- ggplot(dfs, aes(latencies)) + 
    geom_density(aes(fill=sesGroup.y), alpha=0.4) + 
    scale_fill_manual(values = cols, name = 'SES') +
    labs(title="Peak ERF latency", 
         subtitle="Density plot grouped by SES",
         x="Time (s)",
         fill="SES")
(lat.bpd / lat.dens)

# plots of peak amplitudes
auc <- ggplot(dfs, aes(x=oddballCond, y=logauc, fill=hemisphere))
auc.bpd <- auc + 
    geom_boxplot(alpha = 0.5, position = position_dodge(width = 0.9)) +
    facet_grid(. ~ sesGroup.y) +
    scale_color_manual(values = cols, name = 'Hemisphere') +
    labs(title = "Peak ERF acivity",
         subtitle = "grouped by condition & hemisphere over SES",
         x = "Stimulus", y = "AU")

auc.dens <- ggplot(dfs, aes(logauc)) + 
    geom_density(aes(fill=sesGroup.y), alpha=0.4) + 
    scale_fill_manual(values = cols, name = 'SES') +
    scale_color_manual(values = cols) +
    labs(title="Peak ERF activity", 
         subtitle="Density plot grouped by SES",
         x="AU",
         fill="SES")
(auc.bpd / auc.dens)

# Correlation matrices
suppressMessages(library(GGally))

# CDI v Demographics
ggpairs(dfs, aes(color=sesGroup.y, alpha = 0.4),
        columns = c("vocab", "m3l", "ses.y", "age.x", "headSize.x",
                    "birthWeight.x"),
        columnLabels = c("Vocabulary", "M3l", "SES", "Age (d)", "Head sz (cm)", "Weight (oz)"),
        lower = list(
            continuous = wrap("smooth", alpha = 0.4, size=0.2)))

# MEG v Demographics
ggpairs(dfs, aes(color=sesGroup.y, alpha = 0.4),
        columns = c("latencies", "logauc", "ses.y", "age.x", "headSize.x",
                    "birthWeight.x"),
        columnLabels = c("Latency (s)", "Strength (AU)", "SES", "Age (d)", "Head sz (cm)", "Weight (oz)"),
        lower = list(
            continuous = wrap("smooth", alpha = 0.4, size=0.2)))

# Correlation testing (http://www.sthda.com/english/wiki/correlation-test-between-two-variables-in-r)
suppressMessages(library("ggpubr"))

# Vocab v SES
# Shapiro-Wilk normality tests
shapiro.test(ses <- as.numeric(unlist(dfs  %>% select(ses.y))))
ggqqplot((ses), ylab = "SES")

shapiro.test(vocab <- as.numeric(unlist(dfs %>% select(vocab))))
ggqqplot(log(vocab), ylab = "Vocabulary size")

ggscatter(dfs, x = "ses.y", y = "vocab", 
          color = "black", shape = 21, size = 1, # Points color, shape and size
          add = "reg.line", conf.int = TRUE,
          add.params = list(color = "blue", fill = "lightgray"), # Customize reg. line
          cor.coef = TRUE, 
          cor.coeff.args = list(method = "spearman"),
          xlab = "SES", ylab = "Vocabulary size",
          title = "Spearman's rank correlation rho")
cor.test(vocab, ses,  method = "spearman")

# m3l v SES
m3l <- as.numeric(unlist(dfs %>% select(m3l)))
shapiro.test(m3l <- as.numeric(unlist(dfs %>% select(m3l))))
ggqqplot(log(m3l), ylab = "M3l")

ggscatter(dfs, x = "ses.y", y = "m3l", 
          color = "black", shape = 21, size = 1, # Points color, shape and size
          add = "reg.line", conf.int = TRUE,
          add.params = list(color = "blue", fill = "lightgray"), # Customize reg. line
          cor.coef = TRUE, 
          cor.coeff.args = list(method = "spearman"),
          xlab = "SES", ylab = "M3L",
          title = "Spearman's rank correlation rho")
cor.test(m3l, ses,  method = "spearman")

# MEG v CDI
cdi.30 <- dfs %>% filter(cdiAge  %in% c("30") & vocab > 0 & m3l > 0)
ggpairs(cdi.30, 
        aes(color=sesGroup.y, alpha = 0.4),
        columns = c("latencies", "logauc", "vocab", "m3l"),
        columnLabels = c("Latency (s)", "Strength (AU)", "Vocabulary sz", "M3l"),
        lower = list(
            continuous = wrap("smooth", alpha = 0.4, size=0.2)))
# MEG amp v CDI M3L at 30 mo
amp.30 <- as.numeric(unlist(cdi.30 %>% select(logauc)))
shapiro.test(amp.30)
ggqqplot(log(amp.30), ylab = "latency")
m3l.30 <- as.numeric(unlist(cdi.30 %>% select(m3l)))
shapiro.test(m3l.30)
ggqqplot(log(m3l.30), ylab = "M3L")

ggscatter(cdi.30, x = "logauc", y = "m3l", 
          color = "black", shape = 21, size = 1, # Points color, shape and size
          add = "reg.line", conf.int = TRUE,
          add.params = list(color = "blue", fill = "lightgray"), # Customize reg. line
          cor.coef = TRUE, 
          cor.coeff.args = list(method = "spearman"),
          xlab = "ERF strength", ylab = "M3L",
          title = "Spearman's rank correlation rho")

cor.test(m3l.30, amp.30,  method = "spearman")


# RM ANOVA with AFEX
suppressMessages(library(afex))
# CDI VOCAB
(vocab.aov <- aov_ez("subjId", "vocab",
                     dfs %>% filter(cdiAge  %in% c("27", "30") & vocab > 0 & m3l > 0),
                     between = c("sesGroup.y", "gender.x"), 
                     within = c("cdiAge"),
                     anova_table = list(es = "pes", correction = "GG"),
                     return = afex_options("return_aov"),
                     data=)
)
# gender:age post-hoc
library(emmeans)
pairs(emmeans::emmeans(mrt, c("stimulus", "frequency"), by = "task"))
vocab.emms <- emmeans(vocab.aov, ~gender.x:cdiAge)
vocab.pairs <- cld(vocab.emms,
                   alpha=0.05,
                   Letters=letters,  ### Use lower-case letters for .group
                   adjust="bonferroni")  ###  bonferroni-adjusted comparisons
vocab.pairs$.group=gsub(" ", "", vocab.pairs$.group, fixed = T)  
# plot
library(ggrepel)
pd <- position_dodge(width = 0.9)
ggplot(vocab.pairs, aes(x = cdiAge, y = emmean, fill = gender.x, label = .group)) +
    geom_point(shape = 21, size = 4, position = pd) +
    geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL),
                  width = 0.2, size = 0.5, position = pd) +
    theme_bw() +
    theme(axis.title = element_text(face = "bold"),
          axis.text = element_text(face = "bold"),
          plot.caption = element_text(hjust = 0)) +
    ylab("Vocabulary size") +
    scale_fill_manual(name="Gender",values=cols) +
    scale_x_discrete("Age (Months)",
                     labels=c("X27"="27", "X30"="30")) +
    geom_text_repel(position = pd,
                    point.padding=unit(1,'lines'),
                    direction = 'both',
                    segment.size = 0,
                    show.legend = F,
                    color = "grey") +
    ggtitle ("CDI Vocabulary size at 27- and 30-months") +
    labs(caption  = paste0("\nCDI Vocabulary size at ",
                           "27- and 30-months grouped over age and gender. \n",
                           "Plot shows LS Means\u00b195% confidence ",
                           "interval.\n",
                           "Means sharing a letter are ",
                           "not significantly different \n",
                           "(Bonferroni-adjusted comparisons)."), 
         hjust=0.5)
print(vocab.pairs)

# CDI M3L
(m3l.aov <- aov_ez("subjId", "m3l", c("sesGroup.y", "gender.x"), 
                   c("cdiAge"),
                   anova_table = list(es = "pes", correction = "GG"),
                   return = afex_options("return_aov"),
                   data=dfs %>% filter(cdiAge  %in% c("27", "30") & vocab > 0 & m3l > 0))
)
# gender:SES post-hoc
m3l.emms <- emmeans(m3l.aov, ~sesGroup.y:gender.x)
m3l.pairs <- cld(m3l.emms,
                 alpha=0.05,
                 Letters=letters,  ### Use lower-case letters for .group
                 adjust="bonferroni")  ###  bonferroni-adjusted comparisons
m3l.pairs$.group=gsub(" ", "", m3l.pairs$.group, fixed = T)  
# plot
ggplot(m3l.pairs, aes(x = sesGroup.y, y = emmean, fill = gender.x, label = .group)) +
    geom_point(shape = 21, size = 4, position = pd) +
    geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL),
                  width = 0.2, size = 0.5, position = pd) +
    theme_bw() +
    theme(axis.title = element_text(face = "bold"),
          axis.text = element_text(face = "bold"),
          plot.caption = element_text(hjust = 0)) +
    ylab("M3L") +
    scale_fill_manual(name="Gender",values=cols) +
    scale_x_discrete("SES") +
    geom_text_repel(position = pd,
                    point.padding=unit(1,'lines'),
                    direction = 'both',
                    segment.size = 0,
                    show.legend = F,
                    color = "grey") +
    ggtitle ("CDI M3L at 27- and 30-months") +
    labs(caption  = paste0("\nCDI M3L at ",
                           "27- and 30-months grouped over SES and gender. \n",
                           "Plot shows LS Means\u00b195% confidence ",
                           "interval.\n",
                           "Means sharing a letter are ",
                           "not significantly different \n",
                           "(Bonferroni-adjusted comparisons)."), 
         hjust=0.5)
print(m3l.pairs)

# MEG Latency
(latencies.aov <- aov_ez("subjId", "latencies",
                         between = c("sesGroup.y", "gender.x"),
                         within = c("hemisphere", "oddballCond"),
                         anova_table = list(es = "pes", correction = "GG"),
                         return = afex_options("return_aov"),
                         data=dfs))
# sesGroup:hemisphere post-hoc
latencies.emms <- emmeans(latencies.aov, ~sesGroup.y:hemisphere:oddballCond)
latencies.pairs <- cld(latencies.emms,
                       alpha=0.05,
                       Letters=letters,  ### Use lower-case letters for .group
                       adjust="bonferroni")  ###  bonferroni-adjusted comparisons
latencies.pairs$.group=gsub(" ", "", latencies.pairs$.group, fixed = T)  
# plot
ggplot(latencies.pairs, aes(x = hemisphere, y = emmean, fill = oddballCond, label = .group)) +
    geom_point(shape = 21, size = 4, position = pd) +
    geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL),
                  width = 0.2, size = 0.5, position = pd) +
    facet_grid(. ~ sesGroup.y) +
    theme_bw() +
    theme(axis.title = element_text(face = "bold"),
          axis.text = element_text(face = "bold"),
          plot.caption = element_text(hjust = 0)) +
    ylab("Latency (s)") +
    scale_fill_manual(name="Stimulus",values=cols) +
    geom_text_repel(position = pd,
                    point.padding=unit(1,'lines'),
                    direction = 'both',
                    segment.size = 0,
                    show.legend = F,
                    color = "grey") +
    ggtitle ("ERF peak latency") +
    labs(caption  = paste0("Oddball ERF peak latency ",
                           "grouped over SES and hemisphere \n",
                           "Plot shows LS Means\u00b195% confidence ",
                           "interval.\n",
                           "Means sharing a letter are ",
                           "not significantly different \n",
                           "(Bonferroni-adjusted comparisons)."), 
         hjust=0.5)

# MEG amplitude
(auc.aov <- aov_ez("subjId", "auc",
                   transformation = c("log"),
                   between = c("sesGroup.y", "gender.x"),
                   within = c("hemisphere", "oddballCond"),
                   anova_table = list(es = "pes", correction = "GG"),
                   return = afex_options("return_aov"),
                   data=dfs %>% filter(cdiAge  %in% c("27", "30"))))

# sesGroup
auc.emms <- emmeans(auc.aov, ~sesGroup.y)
auc.pairs <- cld(auc.emms,
                 alpha=0.05,
                 Letters=letters,  ### Use lower-case letters for .group
                 adjust="bonferroni")  ###  bonferroni-adjusted comparisons
auc.pairs$.group=gsub(" ", "", auc.pairs$.group, fixed = T)  
# plot
ggplot(auc.pairs, aes(y = emmean, x = sesGroup.y, label = .group)) +
    geom_point(shape = 21, size = 4, position = pd) +
    geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL),
                  width = 0.2, size = 0.5, position = pd) +
    theme_bw() +
    theme(axis.title = element_text(face = "bold"),
          axis.text = element_text(face = "bold"),
          plot.caption = element_text(hjust = 0)) +
    ylab("Strength (AU)") +
    scale_x_discrete("SES") +
    geom_text_repel(position = pd,
                    point.padding=unit(1,'lines'),
                    direction = 'both',
                    segment.size = 0,
                    show.legend = F,
                    color = "grey") +
    ggtitle ("ERF peak activation") +
    labs(caption  = paste0("Oddball ERF peak activation ",
                           "grouped over SES. \n",
                           "Plot shows LS Means\u00b195% confidence ",
                           "interval.\n",
                           "Means sharing a letter are ",
                           "not significantly different \n",
                           "(Bonferroni-adjusted comparisons)."), 
         hjust=0.5)
print(auc.pairs)


