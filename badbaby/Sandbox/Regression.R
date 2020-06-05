

# Regression modelling. See http://r-statistics.co/Linear-Regression.html
# VOCAB over SES
# get dfs labels
dfs.info <- data.frame(attributes(dfs)[c("names")])
# examine lables
tail(dfs.info,8)
par(mfrow=c(1, 2))  # divide graph area in 2 columns
boxplot(dfs$vocab, main="vocab", 
        sub=paste("Outlier rows: ", boxplot.stats(dfs$vocab)$out))  # box plot for 'vocab'
boxplot(dfs$m3l, main="m3l", 
        sub=paste("Outlier rows: ", boxplot.stats(dfs$m3l)$out))  # box plot for 'm3l'
library(e1071)
plot(density(dfs$vocab), main="Density Plot: vocab", ylab="Frequency", 
     sub=paste("Skewness:", round(e1071::skewness(dfs$vocab), 2)))  # density plot for 'vocab'
polygon(density(dfs$vocab), col="red")
plot(density(dfs$m3l), main="Density Plot: m3l", ylab="Frequency", 
     sub=paste("Skewness:", round(e1071::skewness(dfs$m3l), 2)))  # density plot for 'm3l'
polygon(density(dfs$m3l), col="red")
# Scatterplot Matrices from the car Package
library(car)
scatterplotMatrix(~vocab+m3l+ses, data=dfs,
                  main="CDI Vocab & M3L by SES")

# Scatterplot Matrices from the glus Package 
library(gclus)
dta <- dfs[c(3,4,6)] # get data 
dta.r <- abs(cor(dta)) # get correlations
dta.col <- dmat.color(dta.r) # get colors
# reorder variables so those with highest correlation
# are closest to the diagonal
dta.o <- order.single(dta.r) 
cpairs(dta, dta.o, panel.colors=dta.col, gap=.5,
       main="Variables Ordered and Colored by Correlation" )


# Summary of vocab and ses columns, all rows.

dfs.vocab.ses <- subset(dfs, select = c("vocab", "ses"))
summary(dfs.vocab.ses)
# correlation between vocab and ses
cor(dfs.vocab.ses) 
# scatter plot of vocab vs ses
scatter.smooth(x=dfs$ses, y=dfs$vocab, main="vocab ~ SES")  # scatterplot
scatterplot(vocab ~ ses, data = dfs,
            xlab="SES", ylab="Vocabulary", 
            main="VOCAB Scatter Plot", 
            smooth = T, grid = FALSE, frame = FALSE)
library(ggpubr)
ggscatter(dfs, x = "ses", y = "vocab", 
          add = "reg.line", conf.int = TRUE, 
          cor.coef = TRUE, cor.method = "pearson",
          xlab = "SES", ylab = "Vocabulary")

# Linear model
# null model, grouping by subjects but not fixed effects.
Norm1 <-lmer(vocab ~ 1 + (1|subjId),
             data=na.omit(dfs), REML = FALSE)
summary(Norm1)

# Scale pvars
dfs$sses <- scale(dfs$ses)
Norm2 <-lmer(vocab~sses + (1|subjId),
             data=na.omit(dfs),
             REML = FALSE) 
summary(Norm2)


#Summarize and print the results

summary(Norm2) # show regression coefficients table
anova(Norm1, Norm2)
confint(Norm2)
hist(residuals(Norm2))
# OLS regression relies on several assumptions, including that the residuals 
# are normally distributed and homoscedastic, the errors are independent and 
# the relationships are linear.
par(mar = c(4, 4, 2, 2), mfrow = c(1, 1)) #optional
plot(Norm2) 
# The Q-Q plot is a probability plot of the standardized residuals against the 
# values that would be expected under normality. If the model residuals are 
# normally distributed then the points on this graph should fall on the straight 
# line, if they don't, then you have violated the normality assumption.
qqnorm(resid(Norm2))
qqline(resid(Norm2))


# M3L over SES
# Summary of vocab and ses columns, all rows.

dfs.m3l.ses <- subset(dfs, select = c("m3l", "ses"))
summary(dfs.m3l.ses)
# correlation between m3l and ses
cor(dfs.m3l.ses) 
# scatter plot of m3l vs ses
scatter.smooth(x=dfs$ses, y=dfs$m3l, main="m3l ~ SES")  # scatterplot
scatterplot(m3l ~ ses, data = dfs,
            xlab="SES", ylab="M3L", 
            main="M3L Scatter Plot", 
            smooth = T, grid = FALSE, frame = FALSE)
library(ggpubr)
ggscatter(dfs, x = "ses", y = "m3l", 
          add = "reg.line", conf.int = TRUE, 
          cor.coef = TRUE, cor.method = "pearson",
          xlab = "SES", ylab = "M3l")

# Linear model
# null model, grouping by subjects but not fixed effects.
Norm1 <-lmer(m3l ~ 1 + (1|subjId),
             data=na.omit(dfs), REML = FALSE)
summary(Norm1)
# Scale pvars
dfs$sses <- scale(dfs$ses)
Norm2 <-lmer(m3l~sses + (1|subjId),
             data=na.omit(dfs),
             REML = FALSE) 
library(effects)
ee <- Effect(c("sses"),Norm2) 
theme_set(theme_bw())
ggplot(as.data.frame(ee),
       aes(sses,fit))+
    geom_line()+
    ## colour=NA suppresses edges of the ribbon
    geom_ribbon(colour=NA,alpha=0.1,
                aes(ymin=lower,ymax=upper))+
    ## add rug plot based on original data
    geom_rug(data=ee$data,aes(y=NULL),sides="b")

# Summarize and print the results
summary(Norm2) # show regression coefficients table
anova(Norm1, Norm2)
nagelkerke(Norm2, Norm1)
confint(Norm2)
hist(residuals(Norm2))
# OLS regression relies on several assumptions, including that the residuals 
# are normally distributed and homoscedastic, the errors are independent and 
# the relationships are linear.
par(mar = c(4, 4, 2, 2), mfrow = c(1, 1)) #optional
plot(Norm2) 
# The Q-Q plot is a probability plot of the standardized residuals against the 
# values that would be expected under normality. If the model residuals are 
# normally distributed then the points on this graph should fall on the straight 
# line, if they don't, then you have violated the normality assumption.
qqnorm(resid(Norm2))
qqline(resid(Norm2))
