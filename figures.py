"""
Thanks to J. Walton's post [1]---instructions for
pdf-perfect plots from matplotlib; and to Sihan for
preparing the first version of this code.

[1]: jwalton.info/Embed-Publication-Matplotlib-Latex/
"""
import string
import matplotlib.pyplot as plt



# Setting up MatPlotLib fonts and sizes

TEX_PARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize":  10,
    "font.size":       10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}

def use_tex_params(override={}):
    """
    Set up MatPlotLib to use LaTeX for all text.
    """
    plt.rcParams.update(TEX_PARAMS)
    plt.rcParams.update(override)
    

# Creating appropriately sized figures

INCH_TO_PT_TEX = 72.27
GOLDEN_RATIO   = (1 + 5**0.5) / 2


def pt_to_inches(pt):
    return pt / INCH_TO_PT_TEX

def inches_to_pt(inches):
    return inches * INCH_TO_PT_TEX

def pt_to_figsize(pt, fraction=1.0, subplots=(1, 1)):
    """
    Convert a column/textwidth (output of \the\textwidth
    or \the\columnwidth in LaTeX) to inches using the
    golden ratio to determine figure height.
    """
    width = pt_to_inches(pt) * fraction
    height = (width / GOLDEN_RATIO) * (subplots[0] / subplots[1])
    return width, height


def subplots(rows, cols, pt, fraction=1.0, **kwargs):
    figsize = pt_to_figsize(pt, fraction=fraction, subplots=(rows, cols))
    print("Creating", rows, "by", cols, "subplots,", figsize, "inches.")
    return plt.subplots(rows, cols, figsize=figsize, **kwargs)


# Placing titles

def title_subplots(axes, titles=None, **kwargs):
    """
    Add titles to an array of subplots. Default: (a), (b), ...
    Kwargs are passed through to axis.title().
    """
    if titles is None:
        titles = map("({})".format, string.ascii_lowercase)
    # place the titles
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title, **kwargs)