library(readr)
library(plyr)
library(tidyverse)
library(janitor)
library(reshape2)

setwd('~/Github/Projects/badbaby/badbaby/data/')

odd.df <- read_csv('AUC_Oddball_55_lbfgs.csv',
                   col_types = cols(X1 = col_skip()))
str(odd.df)
df1_ <- bind_rows(filter(odd.df, group == '2mos'),
                  filter(odd.df, group == '6mos'))

ind.df <- read_csv('AUC_Individual_55_lbfgs.csv',
                   col_types = cols(X1 = col_skip()))
df2_ <- bind_rows(filter(ind.df, group == '2mos'),
                  filter(ind.df, group == '6mos'))

df <- janitor::clean_names(bind_rows(df1_, df2_), 'snake')
write_csv(df, './AUC_55_lbfgs-df.csv')
