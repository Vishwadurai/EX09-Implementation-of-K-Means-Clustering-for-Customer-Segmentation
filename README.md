# EX 9 Implementation of K Means Clustering for Customer Segmentation
## DATE:
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation.
2. Choosing the Number of Clusters (K).
3. K-Means Algorithm Implementation.
4. Evaluate Clustering Results.
5. Deploy and Monitor.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: D.Vishwa
RegisterNumber:  230500134
*/
```
```
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from  sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data
x=data[['Annual Income (k$)', 'Spending Score (1-100)']]
plt.figure(figsize=(4, 4))
plt.scatter(data['Annual Income (k$)'],x['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
K=3
Kmeans=KMeans(n_clusters=K)
Kmeans.fit(x)
centroids=Kmeans.cluster_centers_
labels=Kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors=['r','g','b']
for i in range(k):
  cluster_points=x[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i], label=f'Cluster{i+1}')
  distances=euclidean_distances(cluster_points, [centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i], radius, color=colors[i], fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, color='k', label='Centroids')
plt.title('K-means Clustring')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```

## Output:
![WhatsApp Image 2024-10-26 at 11 00 10_b4973f84](https://github.com/user-attachments/assets/de32179b-ee81-49b5-a0e8-374a650e5272)
![WhatsApp Image 2024-10-26 at 11 00 34_fda93dca](https://github.com/user-attachments/assets/0e57d6d8-44cf-40c7-9f52-6a3057bc916c)
![WhatsApp Image 2024-10-26 at 11 00 47_31936f96](https://github.com/user-attachments/assets/c4fa17c4-441b-4183-8823-b0c1f79b1c54)
![WhatsApp Image 2024-10-26 at 11 01 09_b264f197](https://github.com/user-attachments/assets/29a2fa52-a0b2-484b-9764-104458625c87)







## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
