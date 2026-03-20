import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

font = font_manager.FontEntry(fname="arial.ttf", name="Arial")
font_manager.fontManager.ttflist.append(font)

mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plot_markersize = 10
opas_markersize = 250
opas_idmarkersize = 200
axis_fontsize = 18
title_fontsize = 20
label_fontsize = 12
legend_fontsize = 15