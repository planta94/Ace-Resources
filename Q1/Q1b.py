import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the data
data = pd.read_csv(r'ingredient.csv')

# Create a plot with 3-by-3 axes
fig, ax  = plt.subplots(ncols=3, nrows=3)
fig.suptitle('Violin Plots for Additives')

# Iterate through the columns of the data and plot a violin plot with the quartiles labelled
# In each of the axes, some key statistical descriptions are included
for i, j in enumerate(data.columns):
    a = sns.violinplot(data=data[j], inner='quartiles', ax=ax.flat[i])
    a.set_xticklabels(j)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    mean = data[j].mean()
    stdev = data[j].std()
    skew = data[j].skew()
    kurtosis = data[j].kurt()
    text = f'''Mean = {mean:.2f}
SD = {stdev:.2f}
Skewness = {skew:.2f}
Kurtosis = {kurtosis:.2f}'''
    a.text(0.05, 0.95, text, transform=a.transAxes, verticalalignment='top', bbox=props)
