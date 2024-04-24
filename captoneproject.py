import os 
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score, confusion_matrix, accuracy_score, classification_report, log_loss
from math import sqrt
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from itertools import product
from time import time

from sklearn.cluster import KMeans, k_means


import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px


# To change scientific numbers to float
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# Increases the size of sns plots
sns.set(rc={'figure.figsize':(8,6)})

data = pd.read_csv("shopping_trends_updated.csv")


if data.index.is_unique==False:
   data = data.drop_duplicates()


# Loop through each column in the DataFrame
for col in data.columns:
    # Check if there are any missing values in the column
    if data[col].isnull().any():
        # Calculate the median of the column
        median = data[col].median()
        # Fill missing values with the median inplace
        data[col].fillna(median, inplace=True)



# Get numerical columns
numerical_columns = data.select_dtypes(include=[np.number]).columns

# Define the threshold for skewness
skew_threshold = 0.5

# Calculate skewness for numerical columns
skewness = data[numerical_columns].skew()

# Loop through each numerical column and perform log transformation if skewness is above the threshold
for col in numerical_columns:
    if np.abs(skewness[col]) > skew_threshold:
        # Add a small constant to handle zero or negative values before log transformation
        min_val = data[col].min()
        if min_val <= 0:
            data[col] = data[col] - min_val + 1
        # Perform log transformation
        data[col] = np.log(data[col])

# Print numerical columns that were transformed
transformed_numerical_columns = [col for col in numerical_columns if np.abs(skewness[col]) > skew_threshold]
print(f"Transformed numerical columns: {transformed_numerical_columns}")



def correct_outliers_columnwise(data, z_score_threshold=3):
    # Initialize an empty DataFrame to store corrected values
    data_corrected = pd.DataFrame(index=data.index, columns=data.columns)

    # Iterate over numerical columns
    for col in data.select_dtypes(include=[np.number]):
        # Calculate z-scores for the column
        z_scores = (data[col] - data[col].mean()) / data[col].std()

        # Identify outliers
        outliers = np.abs(z_scores) > z_score_threshold

        # Correct outliers by replacing them with the median of the column
        median_value = data[col].median()
        data_corrected[col] = np.where(outliers, median_value, data[col])

    return data_corrected



# Apply the correction function to numerical columns individually
data_corrected = correct_outliers_columnwise(data)


features = [col for col in data.columns[1:] if 'ID' not in col.lower()]


raw_data= data.copy()
data = data[features]
categorical_columns = data.select_dtypes(include=['object']).columns
new_data = pd.get_dummies(data, columns = categorical_columns,dtype=int)


X_train = new_data.values
scaler = StandardScaler()
X_std = scaler.fit_transform(X_train)
pca = PCA(n_components=2, random_state = 453)
X_r = pca.fit(X_std).transform(X_std)


def find_best_affinity_propagation_parameters(X_r):
    # Define the range of values for damping
    damping_values = [0.5, 0.6, 0.7, 0.8, 0.9]

    # Initialize variables to store the best silhouette score and its corresponding parameters
    best_score = -1
    best_damping = None

    # Iterate over all damping values
    for damping in damping_values:
        # Fit Affinity Propagation with the current damping value
        af = AffinityPropagation(damping=damping).fit(X_r)

        # Predict cluster labels
        cluster_labels = af.labels_

        # Calculate the silhouette score
        score = silhouette_score(X_r, cluster_labels)

        # Check if the current silhouette score is better than the best score found so far
        if score > best_score:
            best_score = score
            best_damping = damping
            best_labels = cluster_labels
    print("Best Silhouette Score:", best_score)
    print("Best k:", best_damping)
    print("Best labels:", best_labels) 
    return best_damping



def find_best_kmeans_numberofclusters(X_r):
    k_values = range(2, 20)  # for example, trying from 2 to 10 clusters

    best_score = -1
    best_k = None
    best_labels = None

    for k in k_values:
       # Fit KMeans clustering
       kmeans = KMeans(n_clusters=k)
       labels = kmeans.fit_predict(X_r)
    
       # Compute silhouette score
       score = silhouette_score(X_r, labels)
    
       # Update best score and k if needed
       if score > best_score:
         best_score = score
         best_k = k
         best_labels = labels
    print("Best Silhouette Score:", best_score)
    print("Best k:", best_k)
    print("Best labels:", best_labels)     
    return best_k




def afclustering(X_r, find_best_affinity_propagation_parameters):
    # Start the timer
    start_time = time()

    # Find the best affinity propagation parameters
    damping = find_best_affinity_propagation_parameters(X_r)

    # Instantiate and fit the Affinity Propagation model
    af = AffinityPropagation(damping=damping).fit(X_r)
    
    # End the timer
    end_time = time()
    
    # Calculate the running time
    running_time = end_time - start_time
    
    # Get the cluster labels
    cluster_labels = af.labels_

    # Get the number of clusters
    num_clusters = len(af.cluster_centers_indices_)
    
    return cluster_labels, running_time,num_clusters,"Affinity Propagation"




clustering_af,runtime_af,num_clusters_af,name_affinity=afclustering(X_r,find_best_affinity_propagation_parameters)


from sklearn.cluster import KMeans
from time import time

def kmeansclustring(X_r, find_best_kmeans_numberofclusters):
    # Start the timer
    start_time = time()

    # Determine the optimal number of clusters
    n_clusters = find_best_kmeans_numberofclusters(X_r)

    # Instantiate and fit the KMeans model
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_r)
    
    # End the timer
    end_time = time()
    
    # Calculate the running time
    running_time = end_time - start_time
    
    # Get the cluster labels
    cluster_labels = kmeans.labels_

    # Get the number of clusters
    num_clusters = len(set(kmeans.labels_))
    
    return cluster_labels, running_time,num_clusters,"Kmeans"



clustering_kmeans,runtime_kmeans,num_clusters_kmeans,name_kmeans=kmeansclustring(X_r,find_best_kmeans_numberofclusters)



clustring_methods=[clustering_af,clustering_kmeans]
runtimes=[runtime_af,runtime_kmeans]
names=[name_affinity,name_kmeans]
num_clusters=[num_clusters_af,num_clusters_kmeans]



def verification(X_r, clustring_methods, runtimes,num_clusters,names):
    max_silhouette_score = -1
    max_silhouette_runtime = None
    method_label = None
    best_method_name = None

    for method, runtime,num,name in zip(clustring_methods, runtimes,num_clusters,names):
        silhouette_score = metrics.silhouette_score(X_r, method)

        if silhouette_score > max_silhouette_score:
            max_silhouette_score = silhouette_score
            max_silhouette_runtime = runtime
            method_label = method
            best_method_name = name
            num_clusters=num
        elif silhouette_score == max_silhouette_score:
            if runtime < max_silhouette_runtime:
                max_silhouette_runtime = runtime
                method_label = method
                best_method_name = name
                num_clusters=num

    print(max_silhouette_score, max_silhouette_runtime) 
    print(method_label)
    print(best_method_name)
    print(num_clusters)
    return method_label, best_method_name,num_clusters


label,name,num=verification(X_r,clustring_methods,runtimes,num_clusters,names)


# Attachine the clusters back to our initial Dataset that has all the data
raw_data['Clusters'] = label

# Creating a cluster Category
raw_data['Clusters Category'] = 'No Data'
cluster_categories = [f'Cluster {i+1}' for i in range(num)]
for i, category in enumerate(cluster_categories):
    raw_data['Clusters Category'].loc[raw_data['Clusters'] == i] = category



# Define the app
app = dash.Dash(__name__)


# Define the layout of the dashboard
app.layout = html.Div([
    html.H1('Customer Segmentation Dashboard'),
    
    # Dropdown for selecting column
    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': col, 'value': col} for col in raw_data.columns],
        value=raw_data.columns[0],  # Set default value
        multi=False,
        clearable=False
    ),
    
    # Graph for displaying clusters per column
    dcc.Graph(id='cluster-bar-chart'),
])

# Define callback to update graph based on dropdown selection


@app.callback(
    dash.dependencies.Output('cluster-bar-chart', 'figure'),
    [dash.dependencies.Input('column-dropdown', 'value')]
)
def update_cluster_bar_chart(selected_column):
    try:
        # Group data by selected column and clusters, and count occurrences
        cluster_counts = raw_data.groupby([selected_column, 'Clusters']).size().reset_index(name='Count')
        cluster_counts['Clusters'] = cluster_counts['Clusters'] + 1

        # Create bar chart
        fig = px.bar(cluster_counts, x=selected_column, y='Count', color='Clusters', barmode='group')
        fig.update_layout(title=f'{name} Cluster Distribution for {selected_column}')
        return fig
    except Exception as e:
        print(f"Error: {e}")
        return {}


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)