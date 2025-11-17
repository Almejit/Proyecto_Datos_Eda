import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def detectar_outliers(df):

numerical_cols = df.select_dtypes(include=['number']).columns

num_plots = len(numerical_cols)
fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 5 * num_plots))

if num_plots == 1:
    axes = [axes]

for i, col in enumerate(numerical_cols):
    sns.boxplot(y=df[col], ax=axes[i])
    axes[i].set_title(f'Box plot de {col}')
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()