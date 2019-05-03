library(tidyverse)
library(forcats)
library(dplyr)
library(ggplot2)
cdi <- read_tsv("data/Ds-mmn-2mos_cdi_df.tsv") %>% mutate(
  cdiAge = fct_recode(as.factor(cdi$cdiAge),
                      Eighteen = "a",
                      TwentyOne = "b",
                      TwentyFour = "c",
                      TwentySeven = "d",
                      Thirty = "e")
)

ggplot(cdi, aes(x = cdiAge, y = vocab)) + 
  geom_bin2d()
