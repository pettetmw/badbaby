library(readr)
library(tidyverse)
library(janitor)

setwd('~/Github/Projects/badbaby/badbaby/data/')
Df <- read_csv("cdi-meg_55_lbfgs_tidyDf_07012020.csv", 
               col_types = cols(contrasts = col_factor(
                   levels = c("standard-ba", "standard-wa","standard-deviant",
                              "ba-wa")), 
                   gender = col_factor(levels = c("M", "F")), 
                   cdiAge = col_factor(levels = c("18", "21", "24", "27", "30"))))
Df.clean <- janitor::clean_names(Df, 'snake')
str(Df.clean)
write_csv(Df.clean, './cdi-meg_55_lbfgs_tidyDf_07012020_jam.csv')



