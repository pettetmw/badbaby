# Install packages if not already installed
if(!require(tidyverse)){install.packages("tidyverse")}
if(!require(multcomp)){install.packages("multcomp")}
if(!require(nlme)){install.packages("nlme")}

home_dir <- setwd(Sys.getenv('HOME'))
cdi_data_file <- paste(home_dir, 'Github/badbaby/badbaby/data/Ds-mmn-2mos_cdi_df.tsv',
                    sep = "/")
cdi_data <- read.csv(file = cdi_data_file,
                 header=TRUE, sep="\t")
names(cdi_data)

library(tidyverse)
# CDI age vs vocab for SES groups
ggplot(data = cdi_data) +
  geom_boxplot(mapping = aes(x = cdiAge, y=vocab, color = sesGroup))

# CDI age vs m3l for SES groups
ggplot(data = cdi_data) +
  geom_boxplot(mapping = aes(x = cdiAge, y=m3l, color = sesGroup))

# ses distributions
ggplot(data = cdi_data) + 
  geom_point(mapping = aes(x = ses, y = headSize, shape=sesGroup), 
             position = "jitter") 

ggplot(data = cdi_data) +
  geom_histogram(mapping = aes(x = ses, fill = maternalEdu),
                binwidth = 5)
ggplot(data = cdi_data) +
  geom_histogram(mapping = aes(x = ses, fill = paternalEdu),
                 binwidth = 5)
ggplot(data = cdi_data, mapping = aes(x = ses, y = vocab)) + 
  geom_boxplot(mapping = aes(group = cut_width(ses, 5)), varwidth = TRUE)

# Birth weight distributions
ggplot(data = cdi_data, mapping = aes(x = birthWeight)) +
  geom_histogram(mapping = aes(fill=sesGroup), binwidth = 5)
cdi_data %>%
  count(maternalEdu, paternalEdu) %>%
  ggplot(mapping = aes(x = maternalEdu, y = paternalEdu)) + 
    geom_tile(mapping = aes(fill = n))

# RM ANOVA
plot.design(vocab ~ cdiAge * gender * sesGroup, fun=mean, data=cdi_data, main="Group means")
library(nlme)   ## for lme()
library(multcomp)   ## for multiple comparison stuff
## summary() compares each SS with the residual SS and prints out the F-test
summary(Aov.mod <- aov(vocab ~ cdiAge * gender * sesGroup * maternalEdu * maternalHscore
                       + Error(subject/(cdiAge *  maternalEdu + gender + sesGroup)), 
                       data = cdi_data))   
summary(glht(lme_vocab, test = adjusted(type = "bonferroni")))
        
        