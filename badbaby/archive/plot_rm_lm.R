# needed libraries
library(readr)
library(tidyverse)
library(ggpubr)
library(ggplot2)
library(ggsignif)
library(colorspace)
hclwizard(n = 2, gui = "shiny")
## Register custom color palette
ages_colors=colorspace::sequential_hcl(
    n = 2,
    h = c(240, 180),
    c = c(35, 40, 15),
    l = c(35, 92),
    power = c(0.6, 1.1),
    alpha = 0.75
)
# for reproducibility
set.seed(123)

setwd('~/Github/Projects/badbaby/badbaby/data/')

# load data
df <-
    read_csv(
        "AUC_55_lbfgs-RMdf.csv",
        col_types = cols(
            `F1 (2)` = col_skip(),
            `Filter 1` = col_skip(),
            `Filter 2` = col_skip(),
            `Filter 3` = col_skip(),
            id = col_character(),
            cdi_age = col_factor(levels = c('18', '21', '24', '27', '30')),
            gender = col_factor(levels = c("M",
                                           "F")),
            tx = col_factor(levels = c("OB",
                                       "ba", "wa", "b/w"))
        )
    )
str(df)

q <- ggplot2::ggplot(data = dplyr::filter(df, tx == 'b/w'), aes(
    x = cdi_age,
    y = vocab,
    group = id,
    color = ses,
    size = AUC
    )) 

q + geom_point(alpha=.2) +
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



