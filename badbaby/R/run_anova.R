

library(tidyverse)
library(plyr)

df <- select(read.delim("~/Github/badbaby/badbaby/data/Ds-mmn-2mos_cdi_df.tsv"), -X)
glimpse(df)
df$cdiAge <- mapvalues(df$cdiAge, from=letters[1:5], 
                       to=as.character(seq(from=18, to=31, by=3)))
glimpse(df)
cdi.summary <- group_by(df, cdiAge, sesGroup) %>%
    summarise_at(c("vocab", "m3l"), list(mean=mean, sd=sd, min=min, max=max, iqr=IQR), na.rm=TRUE)

pd <- position_dodge(0.1)
ggplot(cdi.summary, aes(x=cdiAge, y=vocab_mean, colour=sesGroup, group=sesGroup)) + 
    geom_errorbar(aes(ymin=vocab_mean-vocab_sd, ymax=vocab_mean+vocab_sd), 
                  colour="black", width=.1, position=pd) +
    geom_line(position=pd) +
    geom_point(position=pd, size=3, shape=21, fill="white") + # 21 is filled circle
    scale_x_discrete("Age (Months)") +
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


#############
# Modelling #
#############
library(nlme)
library(car)
library(rcompanion)

# nlme package
# https://rcompanion.org/handbook/I_09.html
# null model
summary(nlme.null  <-  lme(vocab ~ 1,
                            random=~1|subject,
                            data=df))
# Fixed model
summary(nlme.fixed <- gls(vocab ~ cdiAge * gender * maternalHscore,
                           data=df,
                           method="REML"))
Anova(nlme.fixed)
nagelkerke(nlme.fixed, nlme.null)

# Mixed model
summary(nlme.mixed <- lme(vocab ~ cdiAge * gender * maternalHscore,
                            random=~1|subject,
                            data=df,
                            method="REML"))
Anova(nlme.mixed)
nagelkerke(nlme.mixed, nlme.fixed)
anova(nlme.mixed, nlme.null)

library(emmeans)
marginal=emmeans(nlme.mixed, ~ maternalHscore | cdiAge)
cld(marginal, alpha=0.05, Letters=letters,  ### Use lower-case letters for .group
    adjust="bonferroni",  ### Tukey-adjusted comparisons
    details=TRUE)

Sum=groupwiseMean(vocab ~ maternalHscore | cdiAge,
                  data=df, conf=0.95, digits=3, boot=TRUE, bca=TRUE,
                  traditional=FALSE,
                  percentile=TRUE)

ggplot(Sum, aes(x=maternalHscore, y=Boot.mean)) +
    geom_errorbar(aes(ymin=Bca.lower,
                      ymax=Bca.upper),
                  width=.2, size=0.7, position=pd) +
    geom_point(shape=20, size=4, position=pd) +
    geom_point(position=pd, size=3, shape=21, fill="white") + # 21 is filled circle
    scale_x_discrete("Maternal H-score") +
    ylab("Vocabulary size (Boot-Mean\u00B1CI)") +
    ggtitle("Vocabulary by age") +
    expand_limits(y=0) +                        # Expand y range
    scale_y_continuous(breaks=0:700*200) +         # Set tick separation
    theme_bw() +
    theme(legend.justification=c(1,0),
          legend.position="bottom") +              # Position legend in bottom right
    facet_wrap(~cdiAge)

# lmer package
# https://cran.r-project.org/web/packages/lme4/vignettes/lmer.pdf

library(lme4)

# null model
summary(lmer.null  <-  lmer(vocab ~ 1 + (1 | subject),
                           data=df))
# Mixed model
summary(lmer.mixed <- lmer(vocab ~ cdiAge * gender * maternalHscore + (1 | subject),
                          data=df,
                          REML=T))
Anova(lmer.mixed)
nagelkerke(lmer.mixed, lmer.null)
anova(lmer.null, lmer.mixed)

