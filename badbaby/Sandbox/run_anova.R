
## Read in data and summarize response descriptive
library(tidyverse)

df <- read_csv("~/Github/badbaby/badbaby/data/Ds-mmn-2mos_cdi_df.csv",
               col_types="?ff?????f?ffff??f")
glimpse(df <- select(df, -X1))
cdi.summary <- group_by(df, cdiAge, sesGroup) %>%
    summarise_at(c("vocab", "m3l"), 
                 list(mean=mean, 
                      sd=sd, 
                      min=min, 
                      max=max, 
                      iqr=IQR), 
                 na.rm=TRUE)
write_tsv(cdi.summary, 'cdi_summary.tsv')

####################################################################
# ANOVA                                                            #
# https://egret.psychol.cam.ac.uk/statistics/R/anova.html#overview #
####################################################################
library(ez) # for any of the ez* functions

ezPrecis(df)
ezDesign(df,
         x=sesGroup,
         y=vocab,
         col=cdiAge)
## write out ezStats tables for response variable
(stats <- ezStats(data=df,
                  dv=.(m3l),
                  wid=.(subjId),
                  within=.(cdiAge),
                  between=.(sesGroup)))
write_tsv(stats, 'm3l_stats.tsv')

ezPlot(df, dv=.(m3l), 
       wid=.(subjId),
       within=.(cdiAge),
       between=.(sesGroup),
       x=.(cdiAge),
       x_lab="Age (months)",
       y_lab="Length of utterance",
       split=.(sesGroup), 
       split_lab=c("SES"),
       do_lines=T, 
       do_bars=T,
       levels=list(
           cdiAge=list(
               new_names=c('18', '21','24','27','31')
           ),
           sesGroup=list(
               new_names=c('low', 'high'))),
       print_code=F
)

(anova <- ezANOVA(df, dv=.(m3l), 
                  wid=.(subjId), 
                  within=.(cdiAge),
                  between=.(gender, sesGroup),
                  detailed=T, type="II", return_aov=T))
write_tsv(as_tibble(anova$ANOVA), 'm3l_anova.tsv')

########################
# Post-Hoc Comparisons #
########################
# Table for export results of multiple comparison (post hoc Tukey)
# Source: Modified from https://gist.github.com/cheuerde/3acc1879dc397a1adfb0 
# x is a ghlt object
table_glht <- function(x) {
    pq <- summary(x)$test
    mtests <- cbind(pq$coefficients, pq$sigma, pq$tstat, pq$pvalues)
    error <- attr(pq$pvalues, "error")
    pname <- switch(x$alternativ, less= paste("Pr(<", ifelse(x$df==0, "z", "t"), ")", sep= ""), 
                    greater= paste("Pr(>", ifelse(x$df== 0, "z", "t"), ")", sep= ""), two.sided= paste("Pr(>|",ifelse(x$df== 0, "z", "t"), "|)", sep= ""))
    colnames(mtests) <- c("Estimate", "Std. Error", ifelse(x$df==0, "z value", "t value"), pname)
    return(mtests)
    
}

library(lme4)
library(multcomp)
# lmer model
model <- lmer(m3l ~ cdiAge * sesGroup + (1 | subjId) + (1|cdiAge), data=df)
marginal= emmeans(model, ~ cdiAge:sesGroup)
cld(marginal,
    alpha  = 0.05, 
    Letters= letters,     ### Use lower-case letters for .group
    adjust = "none")     ###  holm-adjusted comparisons

library(rcompanion)
Sum= groupwiseMean(m3l ~ cdiAge + sesGroup,
                    data  = df,
                    conf  = 0.95,
                    digits= 3,
                    traditional= FALSE,
                    percentile = TRUE)

library(ggplot2)
pd= position_dodge(.2)
ggplot(Sum, aes(x=cdiAge,
                y=Mean,
                color= sesGroup)) +
    geom_errorbar(aes(ymin=Percentile.lower,
                      ymax=Percentile.upper),
                  width=.2, size=0.7, position=pd) +
    scale_x_discrete("Age (months)",
                     breaks=as.character(seq(from=1, to=5)),
                     labels=as.character(seq(from=18, to=31, by=3))) +
    geom_point(shape=15, size=4, position=pd) +
    theme_bw() +
    theme(axis.title= element_text(face= "bold")) +
    ylab("m3l (EMM")

    
df$ag <- with(df, interaction(cdiAge, sesGroup))
cell <- lmer(m3l~ag -1 + (1 | subjId) + (1|cdiAge), data=df)
Tukey <- contrMat(table(df$ag), "Tukey")
K1 <- cbind(Tukey, matrix(0, nrow= nrow(Tukey), ncol= ncol(Tukey)))
rownames(K1) <- paste(levels(df$cdiAge)[1], rownames(K1), sep= ":")
K2 <- cbind(matrix(0, nrow= nrow(Tukey), ncol= ncol(Tukey)), Tukey)
rownames(K2) <- paste(levels(df$cdiAge)[2], rownames(K2), sep= ":")
K <- rbind(K1, K2)
colnames(K) <- c(colnames(Tukey), colnames(Tukey))
summary(glht(model, linfct=mcp(cdiAge:sesGroup="Tukey"),
             test= adjusted(type= "holm")))

Anova(lm.m3l)
tmp <- expand.grid(subjId= unique(df$subjId),
                   cdiAge= unique(df$cdiAge),
                   sesGroup= unique(df$sesGroup))
X <- model.matrix(~ cdiAge * sesGroup, data= tmp)
glht(lm.m3l, linfct= X)
predict(lm.m3l, newdata= tmp)

ht.m3l <- glht(lm.m3l, 
               linfct= mcp(age.ses= "Tukey"),
               test= adjusted("hochberg"))
summary(ht.m3l)
summary(gg <- glht(lm.post, 
                   linfct=mcp(age.ses="Tukey"),
                   test= adjusted(type= "holm")))

gg.df <- as.data.frame(table_glht(gg))

write_tsv(as_data_frame(table_glht(gg)), 'm3l_age:ses_glht.tsv')

confint(glht(lm.post, linfct=mcp(mhs.gen.itract="Tukey")))
emmeans(lm.post, pairwise ~ mhs.gen.itract, adjust= "tukey")

summary(lmod <- lmer(m3l ~ 1 + gender * sesGroup
                     + (1 | subjId) + (1 | cdiAge),
                     data=df,
                     REML=T))

library(multcomp)
gg <- glht(lmod, linfct=mcp(cdiAge= "Tukey"), test= adjusted(type= "bonferroni"))
plot(gg)
(write_tsv(as_data_frame(vocab.matHscore.comp), 'vocab_matHscore_comp.tsv'))

plot(TukeyHSD(vocab.anova$aov,
              which="maternalHscore", 
              ordered=T))
library(rcompanion)
library(emmeans)
Sum=groupwiseMean(vocab ~ maternalHscore,
                  data=df, conf=0.95, digits=3, boot=TRUE, bca=TRUE,
                  traditional=FALSE,
                  percentile=TRUE)

ggplot(Sum, aes(x=maternalHscore, y=Boot.mean)) +
    geom_errorbar(aes(ymin=Bca.lower,
                      ymax=Bca.upper),
                  width=.2, size=0.7) +
    geom_point(size=3, shape=21, fill="white") + # 21 is filled circle
    scale_x_discrete("Maternal H-Score",
                     breaks=as.character(seq(from=2, to=5)),
                     labels=as.character(seq(from=2, to=5, by=1)))+
    
    ylab("Vocabulary size (Boot-Mean\u00B1CI)") +
    ggtitle("Vocabulary by maternal H-score") +
    expand_limits(y=200) +                        # Expand y range
    scale_y_continuous(breaks=0:700*200) +         # Set tick separation
    theme_bw()

# gender + gender*maternalHscore effects
Tukey.hsd <- TukeyHSD(vocab.anova$aov, which= c("gender", "gender:maternalHscore"), ordered=T)
par(mfrow=c(1,2))
plot(Tukey.hsd)

Tukey.tbl <- as_tibble(Tukey.hsd[["gender:maternalHscore"]])
Tukey.tbl$comps= attributes(Tukey.hsd[["gender:maternalHscore"]])[["dimnames"]][[1]]
Tukey.tbl
write_tsv(filter(as_data_frame(Tukey.tbl),Tukey.tbl$`p adj`<.05), 'vocab_mhsxgen_comp.tsv')

df$mhs.gen.itract <- interaction(df$gender, df$maternalHscore)
lm.post <- lmer(vocab~-1 + mhs.gen.itract + (1 | subjId) + (1 | cdiAge),
                data=df)
summary(inract.comp <- glht(lm.post, linfct=mcp(mhs.gen.itract="Tukey"),
                            test= adjusted(type= "holm")))
confint(glht(lm.post, linfct=mcp(mhs.gen.itract="Tukey")))
emmeans(lm.post, pairwise ~ mhs.gen.itract, adjust= "tukey")
write_tsv(as_data_frame(table_glht(inract.comp)), 'vocab_mhsxgen_comp.tsv')

Sum=groupwiseMean(vocab ~ gender*maternalHscore,
                  data=df, conf=0.95, digits=3, boot=TRUE, bca=TRUE,
                  traditional=FALSE,
                  percentile=TRUE)

pd <- position_dodge(0.2)
ggplot(Sum, aes(x=maternalHscore, y=Boot.mean, group=gender, shape=gender)) +
    geom_errorbar(aes(ymin=Bca.lower,
                      ymax=Bca.upper),
                  color="black", width=.2, size=0.7, position=pd) +
    geom_point(position=pd, size=3, fill="white") + 
    scale_x_discrete("Maternal H-Score",
                     breaks=as.character(seq(from=2, to=5)),
                     labels=as.character(seq(from=2, to=5, by=1)))+
    ylab("Vocabulary size (Boot-Mean\u00B1CI)") +
    scale_shape(name="gender",
                breaks=c(1,2),
                labels=c("M","F")) +
    ggtitle("Vocabulary by maternal H-score") +
    expand_limits(y=200) +                        # Expand y range
    scale_y_continuous(breaks=0:600*200) +         # Set tick separation
    theme_bw() +
    theme(legend.justification=c(1,0),
          legend.position=c(1,0))



##############
# Regression #
##############

##############################################################################
# lmer package                                                               #
# https://cran.r-project.org/web/packages/lme4/vignettes/lmer.pdf            #
# http://lme4.r-forge.r-project.org/slides/2009-07-01-Lausanne/3SimpleD.pdf  #
##############################################################################

library(lme4)
library(car)
library(rcompanion)

# null model
summary(vocab.null  <-  lmer(vocab ~ 1 + (1 | subjId),
                             REML=T, data=df))
summary(vocab.fixed <- lmer(vocab ~ 1 + gender * sesGroup * maternalHscore + 
                                (1 | subjId),
                            data=df))

Anova(vocab.fixed)
nagelkerke(vocab.fixed, vocab.null)

# Mixed models
summary(vocab.mixed <- lmer(vocab ~ 1 + gender * sesGroup * maternalHscore
                            + (1 | subject) + (1 | cdiAge),
                            data=df,
                            REML=T))
plot(vocab.mixed)
Anova(vocab.mixed)
nagelkerke(vocab.mixed, vocab.null)
anova(vocab.null, vocab.mixed)

summary(vocab.mixed2 <- lmer(vocab ~ 1 + maternalHscore * gender
                             + (1 | subject) + (1 | cdiAge),
                             data=df,
                             REML=T))
plot(vocab.mixed2)
Anova(vocab.mixed2)
nagelkerke(vocab.mixed, vocab.null)
anova(vocab.mixed, vocab.mixed2)

marginal=emmeans(nlme.mixed, ~ maternalHscore | cdiAge)
cld(marginal, alpha=0.05, Letters=letters,  ### Use lower-case letters for .group
    adjust="bonferroni",  ### Tukey-adjusted comparisons
    details=TRUE)

pd <- position_dodge(0.1)
ggplot(cdi.summary, aes(x=cdiAge, y=vocab_mean, colour=sesGroup, group=sesGroup)) + 
    geom_errorbar(aes(ymin=vocab_mean-vocab_sd, ymax=vocab_mean+vocab_sd), 
                  colour="black", width=.1, position=pd) +
    geom_line(position=pd) +
    geom_point(position=pd, size=3, shape=21, fill="white") + # 21 is filled circle
    scale_x_discrete("Age (Months)",
                     breaks=as.character(seq(from=1, to=5)),
                     labels=as.character(seq(from=18, to=31, by=3)))+
    ylab("vocab (M\u00B1SD)") +
    scale_colour_hue(name="SES group",    # Legend label, use darker colors
                     breaks=c("1", "2"),
                     labels=c("Low", "High"),
                     l=40) +                    # Use darker colors, lightness=40
    ggtitle("Mean vocabulary size development by age") +
    expand_limits(y=0) +                        # Expand y range
    scale_y_continuous(breaks=0:700*100) +         # Set tick every 4
    theme_bw() +
    theme(legend.justification=c(1,0),
          legend.position=c(1,0))               # Position legend in bottom right




