# needed libraries
library(readr)
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(pairwiseComparisons)
library(ggsignif)
library(MASS)
library(colorspace)
# hclwizard(n = 2, gui = "shiny")

# seed for reproducibility
set.seed(123)

setwd('~/Github/Projects/badbaby/badbaby/data/')

# load data
df <-
    read_csv(
        "AUC_55_lbfgs-df.csv",
        col_types = cols(
            contrasts = col_factor(
                levels = c("standard-deviant",
                           "ba-wa", "standard-ba", "standard-wa")
            ),
            gender = col_factor(levels = c("M", "F")),
            group = col_factor(levels = c("2mos",
                                          "6mos")),
            cdi_age = col_factor(levels = c("18", "21", "24", "27", "30")),
            subj_id = col_skip()
        )
    ) %>%
    dplyr::filter(ses > 0) # with SES

p <- ggplot2::ggplot(data = df, aes(
    x = cdi_age,
    y = vocab,
    group = subject,
    color = ses,
    size = auc
)) 
p + geom_point(alpha=.2) +
    stat_smooth(
        aes(group=1),
        method = "rlm",
        color = 'black',
        se = TRUE,
        size = .5,
        show.legend = FALSE
    ) +
    labs(
        title = "Vocabulary size",
        subtitle = "robust linear regression using an M estimator ± error",
        x = "SES",
        y = "Vocab size",
        size = "AUC",
        color = "SES"
    ) +
    ggthemes::theme_fivethirtyeight()

df_rm <-
    read_csv(
        "AUC_55_lbfgs-RMdf.csv",
        col_types = cols(
            `Filter 1` = col_skip(),
            `Filter 3` = col_skip(),
            id = col_character(),
            cdi_age = col_factor(levels = c('18', '21', '24', '27', '30')),
            gender = col_factor(levels = c("M",
                                           "F")),
            tx = col_factor(levels = c("OB",
                                       "ba", "wa", "b/w"))
        )
    )
str(df_rm)

q <- ggplot2::ggplot(data = dplyr::filter(df_rm, tx == 'b/w'), aes(
    x = cdi_age,
    y = m3l,
    group = id,
    color = ses,
    size = AUC
)) 

q + geom_point(alpha=.5) +
    stat_smooth(
        aes(group=1),
        method = "rlm",
        color = 'black',
        se = TRUE,
        size = .5,
        show.legend = FALSE
    ) +
    labs(
        title = "M3l over age",
        subtitle = "robust linear regression using an M estimator ± error",
        x = "CDI Age",
        y = "M3l",
        size = "AUC",
        color = "SES"
    ) +
    ggthemes::theme_fivethirtyeight()
    
