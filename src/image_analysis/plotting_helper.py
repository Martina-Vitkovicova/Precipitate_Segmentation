import matplotlib as mpl


def apply_paper_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman"],
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.facecolor": "#ECECF4",
        "figure.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.4,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#BEBEBE",
        "grid.linewidth": 1.6,
        "grid.alpha": 0.4,
    })

