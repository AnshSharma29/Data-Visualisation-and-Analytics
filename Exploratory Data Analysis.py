#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Q12. Segmentation: Clustering (K-Means) 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler


# In[2]:


# Load the dataset
data = pd.read_csv("CWC2023.csv")

# Display the first few rows of the dataset
print(data.head())


# In[3]:


# Preprocessing
# Drop irrelevant columns
data.drop(columns=['Match_No'], inplace=True)

# Encoding categorical variables
label_encoder = LabelEncoder()
data['Toss'] = label_encoder.fit_transform(data['Toss'])
data['Choice'] = label_encoder.fit_transform(data['Choice'])
data['Winner'] = label_encoder.fit_transform(data['Winner'])

# Convert Margin_Runs_or_Wickets to two separate columns for Runs and Wickets
data['Margin_Runs'] = data['Margin_Runs_or_Wickets'].apply(lambda x: int(x[:-1]) if 'R' in x else np.nan)
data['Margin_Wickets'] = data['Margin_Runs_or_Wickets'].apply(lambda x: int(x[:-1]) if 'W' in x else np.nan)
data.drop(columns=['Margin_Runs_or_Wickets'], inplace=True)

# Handling missing values
data.fillna(0, inplace=True)


# In[4]:


# Exploratory Data Analysis (EDA)
# Summary statistics
print(data.describe())

# Correlation matrix
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Distribution of winning teams
plt.figure(figsize=(10, 6))
sns.countplot(x='Winner', data=data)
plt.title('Distribution of Winning Teams')
plt.xlabel('Team')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[5]:


# Clustering (K-Means)
# Selecting features for clustering
X = data[['Toss', 'Choice', 'Innings1_Run', 'Innings1_Balls', 'Innings1_Wickets', 
          'Innings2_Run', 'Innings2_Balls', 'Innings2_Wickets', 'Margin_Runs', 'Margin_Wickets']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()


# In[10]:


# Based on the Elbow method, let's choose the number of clusters as 3
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
data['Cluster'] = kmeans.labels_


# In[12]:


# Visualize clusters
sns.scatterplot(x='Toss', y='Winner', hue='Cluster', data=data, palette='Set1')
plt.title('Clusters based on Toss and Winner')
plt.xlabel('Toss')
plt.ylabel('Winner')
plt.show()


# In[ ]:




