# Title     : badbaby-MMNds_EDA
# Objective : EDA gists
# Created by: ktavabi
# Created on: 03/19/19

suppressMessages(library(tidyverse))
df <- read_csv("Ds-mmn-2mos_cdi_df.csv")

# factor character type columns 
# (https://gist.github.com/ramhiser/93fe37be439c480dc26c4bed8aab03dd)
df <- df %>%
    mutate(
        char_column = sample(letters[1:5], nrow(df), replace = TRUE))
sum(sapply(df, is.character)) # 2
df <- df %>%
    mutate_if(sapply(df, is.character), as.factor)
sapply(df, class)
df <- select(df, c(-X1, -subject))

# summarize data
(df.summary <- group_by(df, sesGroup, gender) %>%
    summarise_at(c("vocab", "m3l"),
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

# plots of vocab > 21mos
p <- ggplot(df %>% filter(cdiAge > 21), aes(gender, vocab))
bdp <- p + geom_boxplot(aes(color = sesGroup),
                        width = 0.5, size = 0.4, position = position_dodge(0.8)) +
                            scale_color_manual(values = mycols, name = 'SES') +
                            xlab("Gender") +
                            ylab("Vocabulary size") +
                            geom_dotplot(aes(color = sesGroup, fill = sesGroup),
                                         binaxis = 'y', stackdir = 'center',
                                         dotsize = .2, position = position_dodge(0.8)) +
                            scale_fill_manual(values = mycols, name = 'SES') +
                            labs(title = "Vocabulary size grouped by gender over SES",
                                 subtitle = "Each dot represents 1 row in source data",
                                 x = "Gender", y = "Vocabulary size")
        
dens <- ggplot(df %>% filter(cdiAge > 21), aes(vocab)) + 
    geom_density(aes(fill=sesGroup), alpha=0.4) + 
    scale_fill_manual(values = mycols, name = 'SES') +
    scale_color_manual(values = mycols) +
    labs(title="Density plot", 
         subtitle="Vocabulary size grouped by SES",
         x="Vocabulary size",
         fill="SES")

(bdp | dens)

# plots of m3l > 21mos
p <- ggplot(df %>% filter(cdiAge > 21), aes(gender, m3l))
bdp <- p + geom_boxplot(aes(color = sesGroup),
                        width = 0.5, size = 0.4, position = position_dodge(0.8)) +
    scale_color_manual(values = mycols, name = 'SES') +
    xlab("Gender") +
    ylab("M3L") +
    geom_dotplot(aes(color = sesGroup, fill = sesGroup),
                 binaxis = 'y', stackdir = 'center',
                 dotsize = .2, position = position_dodge(0.8)) +
    scale_fill_manual(values = mycols, name = 'SES') +
    labs(title = "Vocabulary size grouped by gender over SES",
         subtitle = "Each dot represents 1 row in source data",
         x = "Gender", y = "Vocabulary size")

dens <- ggplot(df %>% filter(cdiAge > 21), aes(m3l)) + 
    geom_density(aes(fill=sesGroup), alpha=0.4) + 
    scale_fill_manual(values = mycols, name = 'SES') +
    scale_color_manual(values = mycols) +
    labs(title="Density plot", 
         subtitle="M3l grouped by SES",
         x="M3l",
         fill="SES")

(bdp | dens)

# Correlation matrices
suppressMessages(library(car))
scatterplotMatrix(~vocab+ses+age+headSize+birthWeight, 
                  data=df %>% filter(cdiAge > 21),
                  main="Vocabulary size SPLOM")

# Correlation testing (http://www.sthda.com/english/wiki/correlation-test-between-two-variables-in-r)
suppressMessages(library("ggpubr"))

# Shapiro-Wilk normality tests
shapiro.test(
    ses <- as.numeric(unlist(df %>% 
                                 filter(cdiAge > 21) %>% 
                                 select(ses))))
ggqqplot((ses), ylab = "SES")

shapiro.test(
    vocab <- as.numeric(unlist(df %>% 
                                   filter(cdiAge > 21) %>% 
                                   select(vocab))))
ggqqplot(log(vocab), ylab = "Vocabulary size")

ggscatter(df %>% filter(cdiAge > 21), x = "ses", y = "vocab", 
          color = "black", shape = 21, size = 1, # Points color, shape and size
          add = "reg.line", conf.int = TRUE,
          add.params = list(color = "blue", fill = "lightgray"), # Customize reg. line
          cor.coef = TRUE, 
          cor.coeff.args = list(method = "spearman"),
          xlab = "SES", ylab = "Vocabulary size",
          title = "Spearman's rank correlation rho")
cor.test(vocab, ses,  method = "spearman")

# m3l
scatterplotMatrix(~m3l+ses+age+headSize+birthWeight, 
                  data=df %>% filter(cdiAge > 21),
                  main="M3l SPLOM")
m3l <- as.numeric(unlist(df %>% 
                               filter(cdiAge > 21) %>% 
                               select(m3l)))
shapiro.test(
    m3l <- as.numeric(unlist(df %>% 
                                   filter(cdiAge > 21) %>% 
                                   select(m3l))))
ggqqplot(log(m3l), ylab = "Vocabulary size")

ggscatter(df %>% filter(cdiAge > 21), x = "ses", y = "m3l", 
          color = "black", shape = 21, size = 1, # Points color, shape and size
          add = "reg.line", conf.int = TRUE,
          add.params = list(color = "blue", fill = "lightgray"), # Customize reg. line
          cor.coef = TRUE, 
          cor.coeff.args = list(method = "spearman"),
          xlab = "SES", ylab = "M3L",
          title = "Spearman's rank correlation rho")
cor.test(m3l, ses,  method = "spearman")

