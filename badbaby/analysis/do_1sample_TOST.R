# needed libraries
library(readr)
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(TOSTER)
library(pairwiseComparisons)
library(ggsignif)
library(colorspace)
# hclwizard(n = 2, gui = "shiny")
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
        "AUC_55_lbfgs-df.csv",
        col_types = cols(
            contrasts = col_factor(
                levels = c("standard-deviant",
                           "ba-wa", "standard-ba", "standard-wa")
            ),
            gender = col_factor(levels = c("M", "F")),
            group = col_factor(levels = c("2mos",
                                          "6mos")),
            subj_id = col_skip()
        )
    ) %>%
    dplyr::filter(ses > 0 & cdi_age == 18) # RM with SES
   

# using `TOSTER` for equivalence tests on crossectional AUC data
tost <- TOSTER::dataTOSTtwo(
    data = dplyr::filter(df, contrasts == 'standard-deviant'),
    deps = 'auc',
    group = 'group',
    desc = TRUE,
    var_equal = F,
    plots = TRUE
)

# # using ggpubr::compare_means for cell comparisons
# Note that, unpaired two-samples t-test can be used only under certain conditions:
# when the two groups of samples (A and B), being compared, are normally distributed. 
# This can be checked using Shapiro-Wilk test.
# and when the variances of the two groups are equal. This can be checked using F-test.
# Pairwise t-test between groups
(
    pairs <-
        compare_means(
            auc ~ group,
            df,
            method = "wilcox.test",
            p.adjust.method = 'none',
            paired = FALSE,
            group.by = 'contrasts') %>% 
        mutate(y.position = .75)
)

q <- ggplot2::ggplot(df, aes(contrasts, auc, fill = group)) +
    geom_boxplot(notch = T, outlier.shape = 4) +
    geom_point(position = position_jitterdodge(), alpha = 0.3)

q + labs(
    title = "Classification reliability over age",
    x = "Condition",
    y = "AUC (au)",
    color = "Age (mos)"
) +
    scale_x_discrete(labels = c('MMN',
                                bquote(ERF[deviants]),
                                bquote(MMN[plosive]),
                                bquote(MMN[aspirative]))) +
    ggthemes::scale_color_fivethirtyeight(labels = c('two', 'six')) +
    ggthemes::theme_clean() +
    stat_compare_means(aes(group = group), vjust = -.5,
                       label = 'p.format')

    

# using `pairwiseComparisons` package to create a dataframe with results
(pairs <-
        pairwise_comparisons(contrasts, auc, paired = F, var.equal = F,
                             p.adjust.method = 'holm',
                             type = 'np', messages = T) %>%
        dplyr::mutate(.data = ., groups = purrr::pmap(.l = list(group1, group2), 
                                                      .f = c)) %>%
        dplyr::arrange(.data = ., group1))

# using `geom_signif` to display results


