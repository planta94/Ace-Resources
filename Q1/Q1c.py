import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

# Read the data
data = pd.read_csv(r'ingredient.csv')

# Normalise the data to [0, 1]
scaler = MinMaxScaler()
data_norm = scaler.fit_transform(data)

# K-means clustering is used to cluster the data
# The elbow method (using sse) and the silhouette method (using sil) are used to determine the optimum number for k
sse = dict()
sil = dict()
for k in range(2, 11):
    kmeans = KMeans(k)
    kmeans.fit(data_norm)
    sse[k] = kmeans.inertia_
    sil[k] = silhouette_score(data, kmeans.labels_)

# The elbow method
plt.figure()
plt.title('The Elbow Method')
plt.xlabel('No. of Clusters')
plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()

# The Silhouette method
plt.figure()
plt.title('The Silhouette Method')
plt.xlabel('No. of Clusters')
plt.ylabel('Silhouette Score')
sns.pointplot(x=list(sil.keys()), y=list(sil.values()))
plt.show()

# Use k = 3
kmeans = KMeans(3)
kmeans.fit(data_norm)

# Create a new datafrane with an additional column of 'CLuster'
# Check the mean values of the additives in each cluster
data_k3 = data.assign(Cluster=kmeans.labels_)
check = data_k3.groupby('Cluster').agg({'a':'mean', 'b':'mean', 'c':'mean', 'd':'mean', 'e':'mean', 'f':'mean', 'g':'mean', 'h':'mean', 'i':'mean'})

# Add the 'Cluster' column to the normalised dataframe
# Create a melted dataframe and arrange them according the additives (attribute)
data_norm = pd.DataFrame(data_norm, columns=data.columns)
data_norm['Cluster'] = data_k3['Cluster']
data_melt = pd.melt(data_norm, id_vars='Cluster', value_vars=data.columns, var_name='Attribute', value_name='Value')

# Use a snakeplot to visualise and identify the difference of the clusters (in terms of additives' distributions)
plt.figure()
plt.title('Snakeplot')
sns.lineplot(data=data_melt, x='Attribute', y='Value', hue='Cluster')
plt.show()

# Calulate the relative importance of each additive in their clusters
cluster_avg = data_k3.groupby(['Cluster']).mean()
population_avg = data.mean()
relative_imp = cluster_avg / population_avg - 1

# PLot relative importance heatmap
plt.figure()
plt.title('Relative Importance of Attributes')
sns.heatmap(data=relative_imp, annot=True, square=True, fmt='.2f', cmap='RdBu_r')
plt.show()

