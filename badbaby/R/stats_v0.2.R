library("afex") # needed for mixed() and attaches lme4 automatically.
afex_options(emmeans_model = "multivariate")  # ANOVAs involving RM factors, follow-up tests based on the multivariate model are generally preferred to univariate follow-up tests.
set_sum_contrasts()  # orthogonal sum-to-zero contrasts
library("emmeans") # emmeans is needed for follow-up tests (and not anymore loaded automatically).
library("multcomp") # for advanced control for multiple testing/Type 1 errors.
library("coefplot")
library("tidyverse")
library("lattice") # for plots
library("latticeExtra") # for combining lattice plots, etc.
lattice.options(default.theme = standard.theme(color = FALSE)) # black and white
lattice.options(default.args = list(as.table = TRUE)) # better ordering
# Change some of the default lattice settings
my.settings <- canonical.theme(color=FALSE)
my.settings[['strip.background']]$col <- "black"
my.settings[['strip.border']]$col<- "black"
library("ggplot2")
library("ggthemes")
library("cowplot")


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
dfs$cdiAge <- as.factor(dfs$cdiAge)
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

## SPLOM
dfs.s <- dfs %>% filter(vocab > 0 & m3l > 0) %>%
    select('auc', 'latencies', 'm3l', 'vocab', 'ses.y')
## https://procomun.wordpress.com/2011/03/18/splomr/
library(solaR)
library(hexbin)
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

# Mixed Modeling
## https://cran.r-project.org/web/packages/afex/vignettes/afex_mixed_example.html
## centering with 'scale()'
## http://www.gastonsanchez.com/visually-enforced/how-to/2014/01/15/Center-data-in-R/
center_scale <- function(x) {
    scale(x, scale = FALSE)
}
dfs <- mutate(dfs, cses=center_scale(ses.y))
dfs <- mutate(dfs, clauc=center_scale(log10(auc)))

## https://rpubs.com/bbolker/4619
pr <- function(m) printCoefmat(coef(summary(m)),digits=3,signif.stars=T)
## by-s random intercepts and by-s random slopes for a plus their correlation
(m1s <- mixed(vocab ~ cdiAge*gender.x*cses + (cdiAge*cses|subjId), 
             dfs, method = "S", 
             control = lmerControl(optCtrl = list(maxfun = 1e6))))
## by-s random intercepts and by-s random slopes for a, but no correlation
(m2s <- mixed(vocab ~ cdiAge*gender.x*cses + (cdiAge*cses||subjId), 
              dfs, method = "S", 
              control = lmerControl(optCtrl = list(maxfun = 1e6))))

## by-s random intercepts
### vocab
pr(m3s <- mixed(vocab ~ cdiAge*cses + (1|subjId), 
                dfs, method = "S", 
                control = lmerControl(optCtrl = list(maxfun = 1e6))))
pr(m3sg <- mixed(vocab ~ cdiAge*sesGroup.y + (1|subjId), 
                 dfs, method = "S", 
                 control = lmerControl(optCtrl = list(maxfun = 1e6))))
coefplot.rxLinMod(m3sg, cex = .5, sort = c("magnitude"))
emm_options(lmer.df = "asymptotic") # also possible: 'satterthwaite', 'kenward-roger'
### vocab ~ cdiAge:sesGroup.y post-hoc
emm_i1 <- emmeans(m3sg, ~ cdiAge*sesGroup.y)
update(pairs(emm_i1), by = NULL, adjust = "holm")
summary(as.glht(update(pairs(emm_i1), by = NULL)), test = adjusted("free"))
emm_i1b <- summary(contrast(emm_i1, by = NULL))
(emm_i1b[,c("estimate", "SE")] <- exp(emm_i1b[,c("estimate", "SE")]))
p1 <- afex_plot(m3sg, x = "cdiAge", trace = "sesGroup.y", 
                dodge = 0.3,
                data_arg = list(
                    position = 
                        ggplot2::position_jitterdodge(
                            jitter.width = 0, 
                            jitter.height = 10, 
                            dodge.width = 0.3  ## needs to be same as dodge
                        ),
                    color = "darkgrey"))

### m3l
pr(m4s <- mixed(m3l ~ cdiAge*gender.x*cses + (1|subjId), 
                dfs, method = "S", 
                control = lmerControl(optCtrl = list(maxfun = 1e6))))
pr(m4sg <- mixed(m3l ~ cdiAge*gender.x*sesGroup.y + (1|subjId), 
                dfs, method = "S", 
                control = lmerControl(optCtrl = list(maxfun = 1e6))))
coefplot.rxLinMod(m4sg, cex = .5, sort = c("magnitude"))
### m3l ~ cdiAge:gender.x:sesGroup.y post-hoc
emm_i2 <- emmeans(m4sg, ~ cdiAge*gender.x*sesGroup.y)
update(pairs(emm_i2), by = NULL, adjust = "holm")
summary(as.glht(update(pairs(emm_i2), by = NULL)), test = adjusted("free"))
emm_i2b <- summary(contrast(emm_i2, by = NULL))
(emm_i2b[,c("estimate", "SE")] <- exp(emm_i2b[,c("estimate", "SE")]))
p2 <- afex_plot(m4sg, "cdiAge", "gender.x", "sesGroup.y", 
                id = "subjId",
                dodge = 0.65,
                data_arg = list(
                    position = 
                        ggplot2::position_jitterdodge(
                            jitter.width = 0, 
                            jitter.height = 10, 
                            dodge.width = 0.65  ## needs to be same as dodge
                        ),
                    color = "darkgrey"),
                emmeans_arg = list(model = "multivariate"))

### latency
(m4s <- mixed(latencies ~ hemisphere*stimulus*gender.x*cses + (1|subjId), 
              dfs, method = "S", 
              control = lmerControl(optCtrl = list(maxfun = 1e6))))
### magnitude
(m4s <- mixed(clauc ~ hemisphere*stimulus*gender.x*cses + (1|subjId), 
              dfs, method = "S", 
              control = lmerControl(optCtrl = list(maxfun = 1e6))))
(m4lrt <- mixed(clauc ~ hemisphere*stimulus*gender.x*cses + (1|subjId), 
              dfs, method = "LRT", 
              control = lmerControl(optCtrl = list(maxfun = 1e6))))
