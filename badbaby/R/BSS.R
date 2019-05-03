## BSS
library(ISLR)
x <- model.matrix( ~ cdiAge + sesGroup + maternalHscore, df)
dim(df)
sum(is.na(df$vocab))
library(leaps)

summary(regfit.full <- regsubsets(vocab~.,data=df))
reg.summary <- summary(regfit.full <- regsubsets(vocab~.,data=df,
                                                 nvmax=37))
names(reg.summary)
reg.summary$rsq
par(mfrow=c(2,2))
plot(reg.summary$rss, xlab="N Vars", ylab="RSS", type="l")
plot(reg.summary$adjr2, xlab="N Vars", ylab="Adj RSq", type="l")
points(which.max(reg.summary$adjr2), 
       reg.summary$adjr2[which.max(reg.summary$adjr2)],
       col="red", cex =2, pch=20)

plot(reg.summary$cp, xlab =" N Vars ", ylab="Cp", type="l")
points(which.min(reg.summary$cp), reg.summary$cp[which.min(reg.summary$cp)],
       col="red", cex =2, pch=20)
plot(reg.summary$bic, xlab =" N Vars ", ylab="BIC", type="l")
points(which.min(reg.summary$bic), reg.summary$bic[which.min(reg.summary$bic)],
       col="red", cex =2, pch=20)
par(mfrow=c(1,1))
plot(regfit.full, scale="r2")
plot(regfit.full, scale="adjr2")
plot(regfit.full, scale="Cp")
plot(regfit.full, scale="bic")
