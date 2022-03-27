"""
Core plotting utilities
-----------------------

Offers flexible functions to create and save line and bar plots for the given
data.
"""

from pathlib import Path
from typing import List, Tuple
from pandas import DataFrame
from seaborn import pointplot, barplot, set as sns_setting
from numpy import median


def line_plot(dest: Path, data: DataFrame, y_col: str, title: str, x_col: str, unit: str, order: List[str]):
    """
    Generate and save a line plot of the given data with the given filters.
    """
    sns_setting(font_scale=1.4)

    # Base plot
    ax = pointplot(data=data, x=x_col, y=y_col, hue='algorithm', estimator=median,
        ci=99, capsize=0.15, palette='deep', hue_order=order, scale=.6)
    fig = ax.get_figure()
    
    # Labels
    fig.suptitle(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(unit, rotation=0)
    ax.yaxis.set_label_coords(0.03, 1.03)

    # Background
    ax.set_facecolor('lightgrey')
    ax.yaxis.grid(True, color='w')

    # Save figure
    fig.set_size_inches((19.20, 10.54))
    fig.savefig(dest, bbox_inches='tight', pad_inches=0.3, dpi=100, format='pdf', transparent=False)
    fig.clf()


def bar_plot(dest: Path, data: DataFrame, y_col: str, title: str, x_col: str, unit: str, order: Tuple[str]):
    """
    Generate and save a bar plot of the given data with the given filters.
    """
    sns_setting(font_scale=1.4)
    
    # Base plot
    ax = barplot(data=data, x=x_col, y=y_col, order=order, estimator=median)
    fig = ax.get_figure()

    # Labels
    fig.suptitle(title)
    ax.set_xlabel("", fontsize=1)
    ax.set_ylabel(unit, rotation=0)
    ax.yaxis.set_label_coords(0.03, 1.03)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

    # Background
    ax.set_facecolor('lightgrey')
    ax.yaxis.grid(True, color='w')

    # Save figure
    fig.set_size_inches((19.20, 10.54))
    fig.savefig(dest, bbox_inches='tight', pad_inches=0.3, dpi=100, format='pdf', transparent=False)
    fig.clf()
