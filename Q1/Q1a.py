import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

data = pd.read_csv(r'ingredient.csv')

desc = data.describe()

# Pearson Correlation Heatmap
corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure()
plt.title('Pearson Correlation Matrix of Various Variables')
sns.heatmap(corr, mask=mask, cmap='RdBu_r', vmax=1, vmin=-1, center=0,
            square=True, annot=True)
plt.tight_layout()
plt.show()

# ANOVA Test
# Make each of the column to be the dependent variable
res = dict()
anova = dict()
for param in data.columns:
    temp = data.drop(param, axis='columns')
    fact = temp.columns
    ano = ols(f'{param} ~ {fact[0]} + {fact[1]} + {fact[2]} + {fact[3]} + {fact[4]} + {fact[5]} + {fact[6]}',
                data=data).fit()
    answer = sm.stats.anova_lm(ano, type=2)
    res[param] = answer
    anova[param] = ano