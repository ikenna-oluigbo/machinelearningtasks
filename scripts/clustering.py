from sklearn.cluster import KMeans  
from sklearn.metrics import silhouette_score 
from tabulate import tabulate
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

from ml_main import file_path, transform_data

(_, _, model_name) = file_path()
(_, two_Dim_embs, node_list) = transform_data()

def silhouette_scores(Y): 
    print(); print('Predicting the number of possible clusters......')
    silhouette_score_list = []; n = 50; i = 2; cluster_list = []
    while i <= n:
        kmeans = KMeans(n_clusters=i).fit(Y)
        cluster_list.append(i)
        silhouette_score_list.append(silhouette_score(Y, kmeans.labels_))
        i += 1 
    for i in range(len(cluster_list)):
        #silhouette_score_table = tabulate([[cluster_list[i], silhouette_score_list[i]]], headers = ['CLUSTER LIST', 'SILHOUETTE SCORE'], tablefmt = 'orgtbl')
        plt.scatter(cluster_list, silhouette_score_list, marker='*', cmap='magma') 
        plt.annotate(cluster_list[i], xy = (cluster_list[i], silhouette_score_list[i]), xytext = (5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.xlabel('CLUSTER LIST'); plt.ylabel('SILHOUETTE SCORE')
        plt.title('SILHOUETTE SCORE FOR SUITABLE NUMBER OF CLUSTERS', color = 'red')
        #with open('./vector_embs/sil_table.txt', 'a') as st:
        #print(silhouette_score_table)
    plt.savefig('./result/silplot_DANE.png')
    plt.close()
    plt.show()
    d = {'CLUSTER LIST':cluster_list, 'SILHOUETTE SCORE':silhouette_score_list}
    df = pd.DataFrame(d)
    silhouette_df = df.loc[df['SILHOUETTE SCORE'] == df['SILHOUETTE SCORE'].values.max()]  #Get Specific items from a column)    
    highest_sil_score = (silhouette_df['SILHOUETTE SCORE'].tolist())[-1]
    opt_cluster = (silhouette_df['CLUSTER LIST'].tolist())[-1]
    print('Optimal Number of Clusters Calculated: ', opt_cluster); print()   
    return (highest_sil_score, opt_cluster)
    
num_clusters = silhouette_scores(two_Dim_embs)[1]
#silhouette_scores(two_Dim_embs)

def build_clusters(Y):
    print('Building Community Clusters on Embeddings......'); print()
    kmeans = KMeans(n_clusters = num_clusters).fit(Y)
    plt.figure(figsize = (10, 10))
    plt.scatter(Y[:, 0], Y[:, 1], c=kmeans.labels_, s = 50, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='*')
    plt.title('Clusters and centroids for: '+str(model_name)+' Model - '+str(num_clusters)+' Clusters.')
    print('Saving file to disk......')
    plt.savefig('./result/clusters_DANE.png')
    plt.close()
    plt.show()
    
build_clusters(two_Dim_embs) 


def distance(Y, Y_first, Y_second, i_centroid, labels):
    distances = [np.sqrt((x-Y_first)**2+(y-Y_second)**2) for (x, y) in Y[labels == i_centroid]]
    return np.mean(distances)


def clusters_centroids(Y):
    print() 
    print('Calculating Centroid Distances for the Optimal', num_clusters, 'Clusters')
    kmeans = KMeans(n_clusters = num_clusters).fit(Y)  
    labels = kmeans.labels_
    centroid_mean_distances = []
    for i, (Y_first, Y_second) in enumerate(kmeans.cluster_centers_):
        mean_distance = distance(two_Dim_embs, Y_first, Y_second, i, labels)
        centroid_mean_distances.append(mean_distance)
    plt.figure(figsize = (10, 10))
    for i in range(len(centroid_mean_distances)):
        plt.plot(centroid_mean_distances)
        plt.annotate(round(centroid_mean_distances[i], 2), xy = (i, centroid_mean_distances[i]), xytext = (0, 0), textcoords='offset points', ha='right', va='bottom')
    
    for i, cmd in enumerate(centroid_mean_distances):
        plt.scatter([], [], cmap = 'viridis', alpha = 0.3, label = 'Cluster ' + str(i) + ': ' + str(cmd))
    focal_mean = np.mean(centroid_mean_distances)
    with open('./embs/focalmean.txt', mode = 'a') as fm:
        print(str(model_name)+' - '+str(focal_mean)+' - '+str(np.var(centroid_mean_distances)), file = fm)
    plt.legend(loc = 'upper left', scatterpoints = 1, frameon = True, labelspacing = 0.3, title = 'Dissimilarity Matrix || '+' Dissimilarity Index = '+str(focal_mean), ncol = 2)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distance to Centroids for Nodes in Clusters')
    plt.title('Centroids Dissimilarity Mean Distance - Number of Clusters = '+str(num_clusters))
    plt.savefig('./result/dissimilarityMatrix_DANE.png')
    plt.close()
    plt.show()
    print('Saving Clusters.')

clusters_centroids(two_Dim_embs)