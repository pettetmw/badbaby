library(tidyverse)

# Read in data
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
# dfs$cdiAge <- as.factor(dfs$cdiAge)
str(dfs) # structure of the data

## participants per condition:
dfs %>% group_by(subjId, oddballCond) %>%
    summarise(n = n_distinct(subjId)) %>%
    ungroup() %>%
    group_by(oddballCond) %>%
    summarise(n = n())

## are all participants in three stimulus conditions?
dfs %>% group_by(subjId) %>%
    summarise(task = n_distinct(stimulus)) %>%
    as.data.frame() %>% 
    {.$task == 3} %>%
    all()

dfs_long <- dfs %>% gather("mag_type", "mag", auc, logauc)
dfs_long <- dfs_long %>% gather("cdi_type", "cdi", vocab, m3l)

library(lattice) # for plots
library(latticeExtra) # for combining lattice plots, etc.
lattice.options(default.theme = standard.theme(color = FALSE)) # black and white
lattice.options(default.args = list(as.table = TRUE)) # better ordering
# Change some of the default lattice settings
my.settings <- canonical.theme(color=FALSE)
my.settings[['strip.background']]$col <- "black"
my.settings[['strip.border']]$col<- "black"

## Density plots of responses
histogram(~cdi|cdi_type, dfs_long, breaks = "Scott", type = "density",
          scale = list(x = list(relation = "free"), y = list(relation = "free")),
          panel = function(x, ...) {
              panel.histogram(x, ...)
              xn <- seq(min(x), max(x), length.out = 100)
              yn <- dnorm(xn, mean(x), sd(x))
              panel.lines(xn, yn, col = "red")
          })
histogram(~mag|mag_type, dfs_long, breaks = "Scott", type = "density",
          scale = list(x = list(relation = "free"), y = list(relation = "free")),
          xlab = "Strength",
          panel = function(x, ...) {
              panel.histogram(x, ...)
              xn <- seq(min(x), max(x), length.out = 100)
              yn <- dnorm(xn, mean(x), sd(x))
              panel.lines(xn, yn, col = "red")
          })
histogram(~ latencies, data = dfs, type = "density", breaks = "Scott",
          xlab = "Latency (s)",
          panel = function(x, ...) {
              panel.histogram(x, ...)
              xn <- seq(min(x), max(x), length.out = 100)
              yn <- dnorm(xn, mean(x), sd(x))
              panel.lines(xn, yn, col = "red")
          })

## CDI spaghetti plots
p <- ggplot(data = dfs %>% filter(vocab > 0 & m3l > 0), 
            aes(x = cdiAge, y = vocab, group = subjId))
p + geom_line() +
    stat_smooth(aes(group = 1)) + 
    stat_summary(aes(group = 1),
                 geom = "point", fun.y = mean, 
                 shape = 17, size = 3) + 
    facet_grid(cols = vars(sesGroup.y))
p + geom_line() + 
    stat_smooth(aes(group = 1), method = "lm", 
                formula = y ~ x * I(x > 1), se = FALSE) + 
    stat_summary(aes(group = 1), fun.y = mean, 
                 geom = "point",  shape = 17, size = 3) + 
    facet_grid(cols = vars(sesGroup.y))

## plot data aggregated across subjects
## CDI ~ cdiAge:sesGroup.y
agg_p <- dfs %>% filter(vocab > 0 & m3l > 0) %>% 
    group_by(subjId, cdiAge, sesGroup.y) %>%
    summarise_at(c("vocab", "m3l"), 
                 list(mean=mean), na.rm = TRUE) %>%
    ungroup()

xyplot(vocab_mean ~ cdiAge:sesGroup.y, agg_p, 
       jitter.x = TRUE, pch = 20, alpha = 0.5,
       main=list(label="Vocabulary size aggregated over CDI age and SES",
                 cex=1.1),
       ylab=list(label="Vocabulary size", cex=0.75),
       xlab=list(label="Age:SES", cex=0.75), 
       scales=list(cex=0.5,
                   alternating=1,   # axes labels left/bottom 
                   tck = c(1,0)),   # ticks only with labels
       grid = "h", 
       panel = function(x, y, ...) {
           panel.xyplot(x, y, ...)
           tmp <- aggregate(y, by = list(x), mean)
           panel.points(tmp$x, tmp$y, pch = 13, cex = 1.2)
       }) + 
    bwplot(vocab_mean ~ cdiAge:sesGroup.y, agg_p, pch="|", do.out = T)

xyplot(m3l_mean ~ cdiAge:sesGroup.y, agg_p, 
       jitter.x = TRUE, pch = 20, alpha = 0.5,
       main=list(label="M3l aggregated over CDI age and SES",
                 cex=1.1),
       ylab=list(label="m3l", cex=0.75),
       xlab=list(label="Age:SES", cex=0.75), 
       scales=list(cex=0.5,
                   alternating = 1,
                   tck = c(1, 0)),
       grid = "h",
       panel = function(x, y, ...) {
           panel.xyplot(x, y, ...)
           tmp <- aggregate(y, by = list(x), mean)
           panel.points(tmp$x, tmp$y, pch = 13, cex =1.5)
       }) + 
    bwplot(m3l_mean ~ cdiAge:sesGroup.y, agg_p, pch="|", do.out = T)

## MEG ~ hemisphere:oddballCond|sesGroup.y
agg_p2 <- dfs %>% group_by(subjId, hemisphere, sesGroup.y, oddballCond) %>%
    summarise_at(c("auc", "latencies"), 
                 list(mean=mean), na.rm = TRUE) %>%
    ungroup()

## logging
xyplot(auc_mean ~ hemisphere:oddballCond|sesGroup.y, agg_p2, 
       jitter.x = TRUE, pch = 20, alpha = 0.5,
       main=list(label="ERF strength over hemisphere, condition and SES",
                 cex=1.1),
       ylab=list(label="log10 magnitude", cex=0.75),
       xlab=list(label="Hemisphere:Condition", cex=0.75), 
       grid = "h",
       scales = list(y = list(log = 10), cex=0.5,
                     alternating = 1,
                     tck = c(1, 0)),
       par.settings = my.settings,
       par.strip.text=list(col="white", font=2),
       panel = function(x, y, ...) {
           panel.xyplot(x, y, ...)
           tmp <- aggregate(y, by = list(x), mean)
           panel.points(tmp$x, tmp$y, pch = 13, cex =1)
       }) + 
    bwplot(auc_mean ~ hemisphere:oddballCond|sesGroup.y, agg_p2, 
           pch="|", do.out = T, scales = list(y = list(log = 10), cex=0.5))

xyplot(latencies_mean ~ hemisphere:oddballCond|sesGroup.y, agg_p2, 
       jitter.x = TRUE, pch = 20, alpha = 0.5, 
       main=list(label="ERF latency over hemisphere, condition and SES",
                 cex=1.1),
       ylab=list(label="Latency (s)", cex=0.75),
       xlab=list(label="Hemisphere:Condition", cex=0.75), 
       grid = "h",
       scales = list(cex=0.5,
                     alternating = 1,
                     tck = c(1, 0)),
       par.settings = my.settings,
       par.strip.text=list(col="white", font=2),
       panel = function(x, y, ...) {
           panel.xyplot(x, y, ...)
           tmp <- aggregate(y, by = list(x), mean)
           panel.points(tmp$x, tmp$y, pch = 13, cex =1)
       }) + 
    bwplot(latencies_mean ~ hemisphere:oddballCond|sesGroup.y, agg_p2, 
           pch="|", do.out = T)


library("hexbin")
## SPLOM
dfs.s <- dfs %>% filter(vocab > 0 & m3l > 0) %>%
    select('auc', 'latencies', 'm3l', 'vocab', 'cses')
## https://procomun.wordpress.com/2011/03/18/splomr/
splom(dfs.s,
      #panel=panel.hexbinplot,
      pch = 8, cex = .2,
      varnames = c("Magnitude (fT/cm)", "Latency (sec)", "M3L",
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
      pscale=0, varname.cex=0.7
)

########################################################################################## Mixed Modeling
## https://cran.r-project.org/web/packages/afex/vignettes/afex_mixed_example.html
## centering with 'scale()'
## http://www.gastonsanchez.com/visually-enforced/how-to/2014/01/15/Center-data-in-R/
########################################################################################
library("afex") # needed for mixed() and attaches lme4 automatically.
afex_options(emmeans_model = "multivariate")  # ANOVAs involving RM factors, follow-up tests based on the multivariate model are generally preferred to univariate follow-up tests.
set_sum_contrasts()  # orthogonal sum-to-zero contrasts

## https://rpubs.com/bbolker/4619
pr <- function(m) printCoefmat(coef(summary(m)),digits=3,signif.stars=T)
# center/scale 
dfs <- mutate(dfs, cses=scale(ses.y, scale = FALSE))
dfs <- mutate(dfs, cmatH=scale(maternalHscore.y, scale = FALSE))
dfs <- mutate(dfs, clauc=scale(log10(auc), scale = FALSE))
dfs <- mutate(dfs, c_cdiAge = cdiAge - 18) 
dfs.f <- filter(dfs, cdiAge  %in% c("18", "21", "24", "27"))
# VOCAB
# by-s random intercepts
pr(vocab.null.is <- mixed(vocab ~ 1 + (1|subjId), 
                          dfs.f, method = "S", 
                          control = lmerControl(optCtrl = list(maxfun = 1e6)),
                          expand_re = TRUE))
# by-s random intercepts and by-s random slopes for a plus their correlation
pr(vocab.null.isa <- mixed(vocab ~ 1 + (c_cdiAge|subjId), 
                           dfs.f, method = "S", 
                           control = lmerControl(optCtrl = list(maxfun = 1e6)),
                           expand_re = TRUE))
# by-s random intercepts and by-s random slopes without their correlation
pr(vocab.null.isa1 <- mixed(vocab ~ 1 + (c_cdiAge|subjId), 
                          dfs.f, method = "S", 
                          control = lmerControl(optCtrl = list(maxfun = 1e6)),
                          expand_re = TRUE))
# Unconditional growth model conveying Age is the best predictor of measure i.e. one way Anova
pr(vocab.isa <- mixed(vocab ~ c_cdiAge + (c_cdiAge|subjId), 
                      dfs.f, method = "S", 
                      control = lmerControl(optCtrl = list(maxfun = 1e6)),
                      expand_re = TRUE))
# Maximal model with by-s random intercepts and by-s random slopes with their correlation
pr(vocab.isa1 <- mixed(vocab ~ c_cdiAge + gender.x + 
                           latencies * oddballCond * hemisphere +
                           logauc * oddballCond * hemisphere +
                           cses * gender.x +
                           cmatH * gender.x +
                           (c_cdiAge|subjId) + 
                           (latencies * oddballCond * hemisphere|subjId), 
                       dfs.f, method = "S", 
                       control = lmerControl(optCtrl = list(maxfun = 1e6)),
                       expand_re = TRUE))

anova(vocab.null.isa1, vocab.isa1, test="F")

library("emmeans") # emmeans is needed for follow-up tests (and not anymore loaded automatically).
library("multcomp") # for advanced control for multiple testing/Type 1 errors.
library("coefplot")

coefplot.rxLinMod(vocab.lmm2, cex = .5, sort = c("natural"))
emm_options(lmer.df = "asymptotic") # also possible: 'satterthwaite', 'kenward-roger'
### vocab ~ cdiAge
vocab.emm <- emmeans(vocab.lmm2, ~ cdiAge)
update(pairs(vocab.emm), by = NULL, adjust = "holm")
summary(as.glht(update(pairs(vocab.emm), by = NULL)), test = adjusted("holm"))
cdiAge.ht <- as.data.frame(confint(as.glht(update(pairs(vocab.emm), 
                                                  by = NULL)))$confint)
cdiAge.ht$Comparison <- rownames(cdiAge.ht)

library("ggthemes")
library("cowplot")
library("ggbeeswarm")
library("ggpubr")
ggplot(cdiAge.ht, 
       aes(x = Comparison, y = Estimate, ymin = lwr, ymax = upr)) +
    geom_errorbar() + geom_point()
cdiAge_emm <- as.data.frame(summary(contrast(vocab.emm, by = NULL)))
cdiAge_emm$Effect <- rownames(cdiAge_emm)
(cdiAge_emm[,c("estimate", "SE")] <- exp(cdiAge_emm[,c("estimate", "SE")]))
(p1 <- afex_plot(vocab.lmm2, x = "cdiAge",
                 dodge = 0.3, 
                 data_geom = ggbeeswarm::geom_beeswarm,
                 data_arg = list(
                     dodge.width = 0.3,  ## needs to be same as dodge
                     cex = 0.2,
                     color = "darkgrey")) +
        labs(x = "Age (mo)", y = "EMM ± 95% ci",
             tag = "A", title = "Vocabulary size",
             subtitle = "Esimated marginal means"))
# Vocab v SES
# Shapiro-Wilk normality tests
shapiro.test(ses <- as.numeric(unlist(select_at(cdi, vars(ses.y)))))
ggqqplot((ses), ylab = "SES")

shapiro.test(vocab <- as.numeric(unlist(select_at(cdi, vars(vocab)))))
ggqqplot(log(vocab), ylab = "Vocabulary size")

ggscatter(cdi, x = "ses.y", y = "vocab", 
          color = "black", shape = 21, size = 1, # Points color, shape and size
          add = "reg.line", conf.int = TRUE,
          add.params = list(color = "blue", fill = "lightgray"), # Customize reg. line
          cor.coef = TRUE, 
          cor.coeff.args = list(method = "spearman"),
          xlab = "SES", ylab = "Vocabulary size",
          title = "Spearman's rank correlation rho")
cor.test(vocab, ses,  method = "spearman")

## M3L by-s random intercepts
pr(m3l.null <- mixed(m3l ~ 1 + (1|subjId), 
                     cdi, method = "S", 
                     control = lmerControl(optCtrl = list(maxfun = 1e6)),
                     expand_re = TRUE))
pr(m3l.lmm1 <- mixed(m3l ~ cses * cdiAge + (1|subjId), 
                     cdi, method = "S", 
                     control = lmerControl(optCtrl = list(maxfun = 1e6)),
                     expand_re = TRUE))
anova(m3l.null, m3l.lmm1, test="F")
AIC(m3l.lmm1[["full_model"]])
pr(m3l.lmm2 <- mixed(m3l ~ gender.x + cses * cdiAge + (1|subjId), 
                       cdi, method = "S", 
                       control = lmerControl(optCtrl = list(maxfun = 1e6)),
                       expand_re = TRUE))
anova(m3l.lmm1, m3l.lmm2, test="F")
AIC(m3l.lmm2[["full_model"]])
anova(m3l.lmm2)
coefplot.rxLinMod(m3l.lmm2, cex = .5, sort = c("natural"))
### m3l ~ cdiAge
m3l.emm <- emmeans(m3l.lmm2, ~ cdiAge)
update(pairs(m3l.emm), by = NULL, adjust = "holm")
summary(as.glht(update(pairs(m3l.emm), by = NULL)), test = adjusted("holm"))
cdiAge.ht <- as.data.frame(confint(as.glht(update(pairs(m3l.emm), 
                                                  by = NULL)))$confint)
cdiAge.ht$Comparison <- rownames(cdiAge.ht)
ggplot(cdiAge.ht, 
       aes(x = Comparison, y = Estimate, ymin = lwr, ymax = upr)) +
    geom_errorbar() + geom_point()
cdiAge_emm <- as.data.frame(summary(contrast(m3l.emm, by = NULL)))
cdiAge_emm$Effect <- rownames(cdiAge_emm)
(cdiAge_emm[,c("estimate", "SE")] <- exp(cdiAge_emm[,c("estimate", "SE")]))
(p1 <- afex_plot(m3l.lmm2, x = "cdiAge",
                 dodge = 0.3, 
                 data_geom = ggbeeswarm::geom_beeswarm,
                 data_arg = list(
                     dodge.width = 0.3,  ## needs to be same as dodge
                     cex = 0.2,
                     color = "darkgrey")) +
        labs(x = "Age (mo)", y = "EMM ± 95% ci",
             tag = "A", title = "M3L",
             subtitle = "Esimated marginal means"))
# m3l v SES
# Shapiro-Wilk normality tests
shapiro.test(ses <- as.numeric(unlist(cdi  %>% select(ses.y))))
ggqqplot((ses), ylab = "SES")

shapiro.test(m3l <- as.numeric(unlist(cdi %>% select(m3l))))
ggqqplot(log(m3l), ylab = "M3L")

ggscatter(cdi, x = "ses.y", y = "m3l", 
          color = "black", shape = 21, size = 1, # Points color, shape and size
          add = "reg.line", conf.int = TRUE,
          add.params = list(color = "blue", fill = "lightgray"), # Customize reg. line
          cor.coef = TRUE, 
          cor.coeff.args = list(method = "spearman"),
          xlab = "SES", ylab = "M3L",
          title = "Spearman's rank correlation rho")
cor.test(m3l, ses,  method = "spearman")

### latency by-s random intercepts
(latency.null <- mixed(latencies ~ 1 + (1|subjId), 
                       dfs, method = "S", 
                       control = lmerControl(optCtrl = list(maxfun = 1e6)),
                       expand_re = True))
(latency.lmm1 <- mixed(latencies ~ gender.x + cses + hemisphere * oddballCond + (1|subjId), 
                       dfs, method = "S", 
                       control = lmerControl(optCtrl = list(maxfun = 1e6)),
                       expand_re = TRUE))
anova(latency.null, latency.lmm1, test="F")
AIC(latency.lmm1[["full_model"]])
anova(latency.lmm1)
coefplot.rxLinMod(latency.lmm1, cex = .5, sort = c("natural"))

### magnitude
(m4s <- mixed(clauc ~ hemisphere*stimulus*gender.x*cses + (1|subjId), 
              dfs, method = "S", 
              control = lmerControl(optCtrl = list(maxfun = 1e6))))
(m4lrt <- mixed(clauc ~ hemisphere*stimulus*gender.x*cses + (1|subjId), 
              dfs, method = "LRT", 
              control = lmerControl(optCtrl = list(maxfun = 1e6))))
