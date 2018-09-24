# -*- coding: utf-8 -*-

""" """

# Authors: Kambiz Tavabi <ktavabi@gmail.com>
# License: MIT

import matplotlib.pyplot as plt


def setup_pyplot(style='ggplot', font_size=12, title_size=14):
    """helper to format pyplot configuration parameters"""
    plt.style.use(style)  # ggplot or seaborn-notebook
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.labelsize'] = font_size
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = font_size
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = font_size
    plt.rcParams['figure.titlesize'] = title_size
    plt.rcParams['grid.color'] = '0.75'
    plt.rcParams['grid.linestyle'] = ':'
    leg_kwargs = dict(frameon=False, columnspacing=0.1, labelspacing=0.1,
                      fontsize=font_size, fancybox=False, handlelength=2.0,
                      loc='best')
    return leg_kwargs


def box_off(ax):
    """helper to format axis tick and border"""
    # Ensure axis ticks only show up on the bottom and left of the plot.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # Remove the plot frame lines.
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    for axis in (ax.get_xaxis(), ax.get_yaxis()):
        for line in [ax.spines['left'], ax.spines['bottom']]:
            line.set_zorder(3)
        for line in axis.get_gridlines():
            line.set_zorder(1)
    ax.grid(True)
