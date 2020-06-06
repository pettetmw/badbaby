library(readr)
library(tidyverse)
library(janitor)

setwd('~/Github/Projects/badbaby/badbaby/data/')

odd.df <- read_csv('AUC_Oddball_55_lbfgs.csv',
                   col_types = cols(X1 = col_skip()))
str(odd.df)
df1 <- odd.df %>% filter(., simmInclude == 1) 
df1_ <- bind_cols(filter(df1, group=='2mos'),
                  filter(df1, group=='6mos'))

ind.df <- read_csv('AUC_Individual_55_lbfgs.csv',
                   col_types = cols(X1 = col_skip()))
str(ind.df)
df2 <- ind.df %>% filter(., simmInclude == 1) 

df2_ <- bind_cols(filter(df2,group=='2mos'),
                  filter(df2, group=='6mos'))

df <- janitor::clean_names(bind_rows(df1_, df2_), 'snake')
write_csv(df, './AUC_55_lbfgs-RMdf.csv')

