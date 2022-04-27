import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import shap
from sklearn.metrics import mean_absolute_percentage_error

# Read Data and Drop the 'HA_Harvested' Column
data = pd.read_csv(r'palm_ffb.csv')
data.set_index(['Date'], inplace=True)
data = data.drop(['HA_Harvested'], axis=1)
data_norm = data.copy()

# Normalise the data to [0, 1] and create a new dataframe of it
param = data.columns
scaler = MinMaxScaler()
data_norm = scaler.fit_transform(data_norm)
data_norm = pd.DataFrame(data_norm, columns=param, index=data.index)

# Time Series Visualisation
plt.figure()
a = 1.5
plt.rcParams['font.size'] = 14
for i in param:
    plt.plot(data_norm[i] + a, label=i + f' + {a}')
    a += 1.5
plt.xticks(np.arange(0, len(data)+1, 12))
plt.xlabel('Date')
plt.ylabel('Normalised Value*')
plt.title('Time Series of Various Variables')
plt.legend()
plt.show()

# Spearman Correlation Heatmap
spear_corr = data.corr(method='spearman')
mask = np.triu(np.ones_like(spear_corr, dtype=bool))
plt.figure()
plt.title('Spearman Correlation Matrix of Various Variables')
sns.heatmap(spear_corr, mask=mask, cmap='RdBu_r', vmax=1, vmin=-1, center=0,
            square=True, annot=True)
plt.tight_layout()
plt.show()

# Kendall Correlation Heatmap
kendall_corr = data.corr(method='kendall')
mask = np.triu(np.ones_like(spear_corr, dtype=bool))
plt.figure()
plt.title('Kendall Correlation Matrix of Various Variables')
sns.heatmap(kendall_corr, mask=mask, cmap='RdBu_r', vmax=1, vmin=-1, center=0,
            square=True, annot=True)
plt.tight_layout()
plt.show()

# Pearson Correlation Heatmap
corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure()
plt.title('Pearson Correlation Matrix of Various Variables')
sns.heatmap(corr, mask=mask, cmap='RdBu_r', vmax=1, vmin=-1, center=0,
            square=True, annot=True)
plt.tight_layout()
plt.show()

# Train an NN model for Correlation Study
# Use SHAP analysis to study the feature importance
plt.figure()
plt.title('SHAP Analysis for NN Model')
X_train = data_norm.iloc[:, :-1]
Y_train = data_norm.iloc[:, -1]
regr = MLPRegressor(max_iter=2000, n_iter_no_change=200, learning_rate='adaptive', learning_rate_init=0.15)
regr.fit(X_train, Y_train)
pred = regr.predict(X_train) * scaler.data_range_[-1] + scaler.data_min_[-1]
regr_error = mean_absolute_percentage_error(data.iloc[:, -1], pred) * 100
explainer = shap.KernelExplainer(regr.predict, X_train)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)
plt.tight_layout()
plt.show()


# Train an linear model for Correlation Study
# Use SHAP analysis to study the feature importance
# The coefficients of the variables are shown
plt.figure()
plt.title('SHAP Analysis for Linear Model')
linregr = LinearRegression()
linregr.fit(X_train, Y_train)
pred = linregr.predict(X_train) * scaler.data_range_[-1] + scaler.data_min_[-1]
linregr_error = mean_absolute_percentage_error(data.iloc[:, -1], pred) * 100
explainer2 = shap.KernelExplainer(linregr.predict, X_train)
shap_values2 = explainer2.shap_values(X_train)
shap.summary_plot(shap_values2, X_train)
plt.tight_layout()
plt.show()
for i, p in enumerate(param[:-1]):
    print(f'The linear coefficient of {p} is {linregr.coef_[i]:.4f}')
