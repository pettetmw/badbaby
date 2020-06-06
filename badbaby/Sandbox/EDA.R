# This is code to carry out EDA of infant mismatch negativity paradigm. 
# author = Kambiz Tavabi
# copyright = Copyright 2018, Seattle, Washington
# license = MIT
# version = 0.1.0
# maintainer = Kambiz Tavabi
# email = ktavabi@uw.edu
# status = Development

library(tidyverse)
##############
# Read in data
dfs <- merge(read_csv("Ds-mmn-2mos_meg_df.csv"),
             read_csv("Ds-mmn-2mos_cdi_df.csv"),
             by = "subjId") %>% 
    select(c(subjId, stimulus, hemisphere, oddballCond, gender.x, age.x,
             headSize.x, birthWeight.x, hemisphere, auc, latencies, cdiAge, 
             m3l, vocab, ses.y, sesGroup.y, maternalHscore.y))

#########
# Wrangle
# factorize columns 
# (https://gist.github.com/ramhiser/93fe37be439c480dc26c4bed8aab03dd)
dfs <- dfs %>%
    mutate(hemisphere = as.character(hemisphere),
           char_column = sample(letters[1:5], nrow(dfs), replace = TRUE))
sum(sapply(dfs, is.character))
dfs <- dfs %>%
    mutate_if(sapply(dfs, is.character), as.factor)
sapply(dfs, class)
# TODO Gaussian with log transform of M3l & VOCAB or Gamma (glmer:family=Gamma(link=log))
# center scale vars
dfs <- mutate(dfs, c_ses = scale(ses.y, scale = F))  
dfs <- mutate(dfs, c_maths = scale(maternalHscore.y, scale = F))
dfs <- mutate(dfs, c_logAuc = scale(log10(auc), scale = F))
dfs <- mutate(dfs, c_latencies = scale(latencies, scale = F))
dfs <- mutate(dfs, c_vocab = scale(vocab, scale = F))  
dfs <- mutate(dfs, c_m3l = scale(m3l, scale = F))  
dfs <- mutate(dfs, c_cdiAge = scale(cdiAge, center = 24))  # center CDI at 12

str(dfs) # structure of the data

# N-participants per condition:
dfs %>% group_by(subjId, oddballCond) %>%
    summarise(n = n_distinct(subjId)) %>%
    ungroup() %>%
    group_by(oddballCond) %>%
    summarise(n = n())

# Are all participants in three stimulus conditions?
dfs %>% group_by(subjId) %>%
    summarise(task = n_distinct(stimulus)) %>%
    as.data.frame() %>% 
    {.$task == 3} %>%
    all()

#####
library(lattice) # for plots
library(latticeExtra) # for combining lattice plots, etc.
lattice.options(default.theme = standard.theme(color = FALSE)) # black and white
lattice.options(default.args = list(as.table = TRUE)) # better ordering
# Change some of the default lattice settings
my.settings <- canonical.theme(color = FALSE)
my.settings[['strip.background']]$col <- "black"
my.settings[['strip.border']]$col<- "black"

###############
# Density plots
dfs.long <-gather(dfs, "type", "response", vocab, c_vocab)
histogram(~response|type, dfs.long, breaks = "Scott", type = "density",
          scale = list(x = list(relation = "free"), y = list(relation = "free")),
          xlab = "Vocabulary size",
          panel = function(x, ...) {
              panel.histogram(x, ...)
              xn <- seq(min(x), max(x), length.out = 100)
              yn <- dnorm(xn, mean(x), sd(x))
              panel.lines(xn, yn, col = "red")
          })

dfs.long <- gather(dfs, "type", "response", m3l, c_m3l)
histogram(~response|type, dfs.long, breaks = "Scott", type = "density",
          scale = list(x = list(relation = "free"), y = list(relation = "free")),
          xlab = "M3L",
          panel = function(x, ...) {
              panel.histogram(x, ...)
              xn <- seq(min(x), max(x), length.out = 100)
              yn <- dnorm(xn, mean(x), sd(x))
              panel.lines(xn, yn, col = "red")
          })

dfs.long <- gather(dfs, "type", "response", auc, c_logAuc)
histogram(~response|type, dfs.long, breaks = "Scott", type = "density",
          scale = list(x = list(relation = "free"), y = list(relation = "free")),
          xlab = "Strength",
          panel = function(x, ...) {
              panel.histogram(x, ...)
              xn <- seq(min(x), max(x), length.out = 100)
              yn <- dnorm(xn, mean(x), sd(x))
              panel.lines(xn, yn, col = "red")
          })

dfs.long <- gather(dfs, "type", "response", latencies, c_latencies)
histogram(~response|type, dfs.long, type = "density", breaks = "Scott",
          xlab = "Latency (sec.)",
          scale = list(x = list(relation = "free"), y = list(relation = "free")),
          panel = function(x, ...) {
              panel.histogram(x, ...)
              xn <- seq(min(x), max(x), length.out = 100)
              yn <- dnorm(xn, mean(x), sd(x))
              panel.lines(xn, yn, col = "red")
          })
################
# spaghetti plot
library(ggthemes)
sp <- ggplot(data = dfs, aes(x = c_cdiAge, y = m3l, group = subjId, color = sesGroup.y))
sp + geom_line(size = .8, alpha = .5) +
    stat_smooth(aes(group = 1), method = "lm", se = TRUE, 
                size = 1, show.legend = FALSE) +
    stat_summary(aes(group = 1), geom = "point", fun.y = mean, shape = 18, 
                 size = 3, show.legend = FALSE) +
    labs(title = "Language outcome", 
         subtitle = "Regression line ± error",
         x = "Age (in months)",
         y = "Vocabulary size",
         color = "SES") +
    theme_classic() + 
    scale_fill_manual(values = c("#0EF1F1", "#EC7D7B")) +
    theme(axis.title.y = element_text(face = "bold", angle=90),
          axis.title.x = element_text(face = "bold"))


###########
# Box plots
bx <- ggplot(dfs, aes(x = cdiAge, y = vocab, fill = factor(cdiAge))) 
bx +  geom_boxplot(notch = TRUE, outlier.colour = "#E60704", outlier.shape = 8,
                   outlier.size = 4, alpha = .6) + 
    stat_summary(geom = "point", fun.y = mean, shape = 18, 
                 size = 3, show.legend = FALSE) +
    geom_jitter(shape = 20, alpha = .2, position = position_jitter(0.2)) +
    labs(title = "Language outcome", 
         x = "Age (in months)",
         y = "Vocabulary size",
         color = "SES") +
    theme_classic() + 
    theme(axis.title.y = element_text(face = "bold", angle = 90),
          axis.title.x = element_text(face = "bold"))


#####
# TODO compute pairwise correlation coeffs between CDI measures at different ages.
library("hexbin")
########
# SPLOM
dfs.s <- dfs %>% filter(cdiAge == 30) %>%
    select('c_logAuc', 'latencies', 'm3l', 'vocab', 'c_ses')
# https://procomun.wordpress.com/2011/03/18/splomr/
splom(dfs.s,
      #panel=panel.hexbinplot,
      pch = 8, cex = .2,
      varnames = c("Log10 Amplitude", "Latency (sec)", "M3L",
                   "Vocabulary\nsize", "SES"),
      colramp=BTC,
      diag.panel = function(x, ...){
          yrng <- current.panel.limits()$ylim
          d <- density(x, na.rm=TRUE)
          d$y <- with(d, yrng[1] + 0.95 * diff(yrng) * y / max(y) )
          panel.lines(d)
          diag.panel.splom(x, ...)
      },
      lower.panel = function(x, y, ...){
          panel.hexbinplot(x, y, ...)
          panel.loess(x, y, ..., col = 'red')
      },
      pscale = 0, varname.cex=0.7,
      xlab = "Scatter Matrix at 30 months"
)
