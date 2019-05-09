# Title     : Ds-mmn-2mos-meg-eda
# Objective : MMN-2mos MEG EDA
# Created by: ktavabi
# Created on: 03/19/19

suppressMessages(library(tidyverse))
df <- read_csv("Ds-mmn-2mos_meg_df.csv")

# factor columns (https://gist.github.com/ramhiser/93fe37be439c480dc26c4bed8aab03dd)
df <- df %>%
    mutate(
        hemisphere = as.character(hemisphere),
        char_column = sample(letters[1:5], nrow(df), replace = TRUE))
sum(sapply(df, is.character)) # 2
df <- df %>%
    mutate_if(sapply(df, is.character), as.factor)
sapply(df, class)
df <- select(df, c(-X1, -subject))

# summarize data
(df.summary <- group_by(df, sesGroup, oddballCond) %>%
    summarise_at(c("latencies", "auc"),
                 list(
                     mean = mean,
                     sd = sd,
                     min = min,
                     max = max,
                     iqr = IQR
                 ),
                 na.rm = TRUE))

suppressMessages(library(ggplot2))
suppressMessages(library(patchwork))

mycols <- c("#2E9FDF", "#FC4E07")

# plots of latencies
p <- ggplot(df, aes(oddballCond, latencies))
bdp <- p + geom_boxplot(aes(color = sesGroup),
                        width = 0.5, size = 0.4, position = position_dodge(0.8)) +
                            scale_color_manual(values = mycols, name = 'SES') +
                            xlab("Condition") +
                            ylab("Latency (sec)") +
                            geom_dotplot(aes(color = sesGroup, fill = sesGroup),
                                         binaxis = 'y', stackdir = 'center',
                                         dotsize = .2, position = position_dodge(0.8)) +
                            scale_fill_manual(values = mycols, name = 'SES') +
                            labs(title = "Latencies grouped by Condition over SES",
                                 subtitle = "Each dot represents 1 row in source data",
                                 x = "Condition", y = "Latency (sec)")
        
dens <- ggplot(df, aes(latencies)) + 
    geom_density(aes(fill=sesGroup), alpha=0.4) + 
    scale_fill_manual(values = mycols, name = 'SES') +
    scale_color_manual(values = mycols) +
    labs(title="Density plot", 
         subtitle="Latencies grouped by SES",
         x="Latency (sec)",
         fill="SES")

(bdp | dens)

# plots of AUC
df$logauc = log(df$auc)
p <- ggplot(df, aes(oddballCond, logauc))
bdp <- p + geom_boxplot(aes(color = sesGroup),
                        width = 0.5, size = 0.4, position = position_dodge(0.8)) +
    scale_color_manual(values = mycols, name = 'SES') +
    xlab("Condition") +
    ylab("Latency (sec)") +
    geom_dotplot(aes(color = sesGroup, fill = sesGroup),
                 binaxis = 'y', stackdir = 'center',
                 dotsize = .2, position = position_dodge(0.8)) +
    scale_fill_manual(values = mycols, name = 'SES') +
    labs(title = "Magnitude grouped by Condition over SES",
         subtitle = "Each dot represents 1 row in source data",
         x = "Condition", y = "Log Magnitude")

dens <- ggplot(df, aes(logauc)) + 
    geom_density(aes(fill=sesGroup), alpha=0.4) + 
    scale_fill_manual(values = mycols, name = 'SES') +
    scale_color_manual(values = mycols) +
    labs(title="Density plot", 
         subtitle="Magnitude grouped by SES",
         x="Log Magnitude",
         fill="SES")

(bdp | dens)

# summarize response variables for deviants grouped by SES/Hem
df %>% 
    filter(oddballCond == 'deviant') %>% 
    group_by(sesGroup, hemisphere) %>%
    summarise_at(c("latencies", "auc"),
                 list(
                     mean = mean,
                     sd = sd,
                     min = min,
                     max = max,
                     iqr = IQR
                 ),
                 na.rm = TRUE)

# Plot response variables for deviants
p <- ggplot(df %>% filter(oddballCond == 'deviant') , aes(hemisphere, latencies))
bdp <- p + geom_boxplot(aes(color = sesGroup),
                        width = 0.5, size = 0.4, position = position_dodge(0.8)) +
    scale_color_manual(values = mycols, name = 'SES') +
    xlab("Condition") +
    ylab("Latency (sec)") +
    geom_dotplot(aes(color = sesGroup, fill = sesGroup),
                 binaxis = 'y', stackdir = 'center',
                 dotsize = .2, position = position_dodge(0.8)) +
    scale_fill_manual(values = mycols, name = 'SES') +
    labs(title = "Peak deviant latency \ngrouped by hemisphere over SES",
         subtitle = "Each dot represents 1 row in source data",
         x = "Hemisphere", y = "Latency (sec)")

dens <- ggplot(df %>% filter(oddballCond == 'deviant'), aes(latencies)) + 
    geom_density(aes(fill=sesGroup), alpha=0.4) + 
    scale_fill_manual(values = mycols, name = 'SES') +
    scale_color_manual(values = mycols) +
    labs(title="Density plot", 
         subtitle="Peak deviant latency grouped by SES",
         x="Latency (sec)",
         fill="SES")

(bdp | dens)

p <- ggplot(df %>% filter(oddballCond == 'deviant') , aes(hemisphere, logauc))
bdp <- p + geom_boxplot(aes(color = sesGroup),
                        width = 0.5, size = 0.4, position = position_dodge(0.8)) +
    scale_color_manual(values = mycols, name = 'SES') +
    xlab("Condition") +
    ylab("Log magnitude") +
    geom_dotplot(aes(color = sesGroup, fill = sesGroup),
                 binaxis = 'y', stackdir = 'center',
                 dotsize = .2, position = position_dodge(0.8)) +
    scale_fill_manual(values = mycols, name = 'SES') +
    labs(title = "Log deviant ERF magnitude \ngrouped by hemisphere over SES",
         subtitle = "Each dot represents 1 row in source data",
         x = "Hemisphere", y = "Log magnitude")

dens <- ggplot(df %>% filter(oddballCond == 'deviant'), aes(logauc)) + 
    geom_density(aes(fill=sesGroup), alpha=0.4) + 
    scale_fill_manual(values = mycols, name = 'SES') +
    scale_color_manual(values = mycols) +
    labs(title="Density plot", 
         subtitle="Log deviant ERF magnitude grouped by SES",
         x="Log magnitude",
         fill="SES")
(bdp | dens)

# Correlation matrices
suppressMessages(library(car))
scatterplotMatrix(~latencies+ses+age+headSize+birthWeight, 
                  data=df %>% filter(oddballCond == 'deviant'),
                  main="Peak deviant latency SPLOM")
scatterplotMatrix(~logauc+ses+age+headSize+birthWeight, 
                  data=df %>% filter(oddballCond == 'deviant'),
                  main="Deviant ERF magnitude SPLOM")

# Correlation testing (http://www.sthda.com/english/wiki/correlation-test-between-two-variables-in-r)
suppressMessages(library("ggpubr"))
# Shapiro-Wilk normality test
shapiro.test(
    deviant.latencies <- as.numeric(unlist(df %>% 
                                               filter(oddballCond == 'deviant') %>% 
                                               select(latencies))))
ggqqplot((deviant.latencies), ylab = "Deviant ERF latencies (sec)")
shapiro.test(
    deviant.auc <- as.numeric(unlist(df %>% 
                                         filter(oddballCond == 'deviant') %>% 
                                         select(logauc))))

dfs <- merge(read_csv("Ds-mmn-2mos_meg_df.csv"),
             read_csv("Ds-mmn-2mos_cdi_df.csv"),
             by = "subjId") %>% 
    select(c(subjId, subject.x, stimulus, hemisphere, oddballCond,
             hemisphere, auc, latencies, cdiAge, m3l, vocab, ses.y, maternalHscore.y))
dfs$logauc = log(dfs$auc)
scatterplotMatrix(~latencies+m3l+vocab+ses.y, 
                  data=dfs %>% filter(oddballCond == 'deviant' & cdiAge > 24),
                  main="Peak deviant latency SPLOM")
scatterplotMatrix(~logauc+m3l+vocab+ses.y, 
                  data=dfs %>% filter(oddballCond == 'deviant' & cdiAge > 24),
                  main="Deviant ERF magnitude SPLOM")



