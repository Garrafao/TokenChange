#!/usr/bin/env python
# coding: utf-8

import os 
import warnings
warnings.filterwarnings("ignore")
from utils_ import Space
from docopt import docopt
import logging
import time
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn import preprocessing
import random
from scipy.spatial import distance
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import sys
from sklearn.cluster import AgglomerativeClustering

def main():

    # Get the arguments
    args = docopt("""

    Usage:
        Clustering.py  <pathVectors> <pathTestSentences> <outPathLabels> <outPathResults> <initializationType> <numberClusters> <clustering> 
        Clustering.py  <pathTestSentences> <initializationType> <numberClusters> <clustering>
        
    Arguments:
       
        <pathVectors> = Path to the vectors
        <pathTestSentences> = Path to the test sentecens that contain the gold clustering, if no performance is needed set to 0
        <outPathLabels> = Path to store the labels
        <outPathResults> = path to store the performance in, if no performance is needed set to 0 
        <initializationType> = "gaac" for precalculated initialization, else random. (Only for kmeans used)
        <numberClusters> = Number of desired clusters, if 0 than its calculated by sillhouette
        <clustering> = Either "hierarchical" or "kmeans"

    
    """)
    
    pathVectors = args['<pathVectors>']
    pathTestSentences = args['<pathTestSentences>']
    initializationType = args['<initializationType>']
    numberClusters = int(args['<numberClusters>'])
    outPathLabels = args['<outPathLabels>']
    outPathResults = args['<outPathResults>']
    clustering = args['<clustering>']
    

    if len(sys.argv) == 5:
        pathVectors = "Files/Vectors/SecondOrder/Vectors.npz"
        outPathLabels = "Files/Clustering/cluster_labels.csv"
        outPathResults = "Files/Clustering/cluster_scores.csv"

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.CRITICAL)
    print("")
    start_time = time.time()  
    logging.critical("Clustering start") 

    #Load vectors
    inSpace = Space(path=pathVectors)
    loaded_contextVectorList_sparse=inSpace.matrix

    
    if pathTestSentences != "0":
    #Get gold clustering if exists
        testSentences=[]
        gold=[]
        with open(pathTestSentences, 'r') as csvFile:
            reader = csv.DictReader(csvFile, delimiter="\t")
            for row in reader:
                testSentences.append(dict(row))   
        for dic in testSentences:
                gold.append(int(dic['cluster']))
            
    if numberClusters == 0:
        #Calculate silhouette score for eaach number of clusters
        range_n_clusters = [2,3,4,5,6,7,8,9,10]
        maxIndex=0
        maxValue=0
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(loaded_contextVectorList_sparse.toarray())
            silhouette_avg = silhouette_score(loaded_contextVectorList_sparse.toarray(), cluster_labels)
            if maxValue <=silhouette_avg:
                maxValue=silhouette_avg
                maxIndex=n_clusters
            numberClusters = maxIndex

      
    
    if clustering == "hierarchical":
        clustering = AgglomerativeClustering(n_clusters=numberClusters).fit(loaded_contextVectorList_sparse.toarray())
        label=clustering.labels_        
    
    else:
        
        if initializationType == "gaac":

            #Calculate GAAC on sample vectors for initial centroids
            testList=[]
            size = min(len(loaded_contextVectorList_sparse.toarray()), 50 )
            randoms=random.sample(range(0, len(loaded_contextVectorList_sparse.toarray())), size)
            for i in randoms: 
                testList.append(loaded_contextVectorList_sparse[i].toarray()[0])   
            initialCentroids=preprocessing.normalize(gaac(testList, numberClusters), norm='l2')

            #Calculate kmeans    
            centroid, label = kmeans2(loaded_contextVectorList_sparse.toarray(),
                                                            initialCentroids , 5, minit='matrix')

        else:
            #Calculate kmeans    
            centroid, label = kmeans2(loaded_contextVectorList_sparse.toarray(),
                                                            numberClusters , 5, minit='points')

    if outPathResults != "0":
        filename = os.path.splitext(os.path.basename(pathTestSentences))[0]

        ADJ=[filename, "ADJ", (round(adjusted_rand_score(gold, label),3)) ]
        ACC=[filename, "ACC", cluster_accuracy(np.array(gold), np.array(label)) ]  
  
        with open(outPathResults, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([ADJ, ACC])    

        #Show results 
        print("")
        print(filename)
        print("")
        print("Adjusted rand index:")
        print(round(adjusted_rand_score(gold, label),3))
        print("Accuracy:")
        print(cluster_accuracy(np.array(gold), np.array(label)))
        print("")
        #plotClusters(loaded_contextVectorList_sparse.toarray(), gold, label)                                  

    #Save labels
    with open(outPathLabels, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([label])    
    logging.critical("Clustering end") 
    logging.critical("--- %s seconds ---" % (time.time() - start_time))
    print("")
    

#Calculates and returns the accuracy for two lists of labels    
def cluster_accuracy(y_true, y_pred):
    # compute confusion matrix
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # Find best mapping between cluster labels and gold labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    #return result
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)
  

#Calulates the clustering (labels) for given vectors and a number of desired clusters 
def gaac(vectors, limit):

    #Put each vector(its ID) in an individual cluster
    clusters=[]
    for i in range(0, len(vectors)):
        clusters.append([i])

    #Compute pairwise distance of all pairs of vectors
    distances = np.zeros(shape=(len(vectors), len(vectors)))
    for i in range(0, len(vectors)):
        for j in range(0, len(vectors)):
            distances[i,j]=distance.cosine(vectors[i], vectors[j])

    #Search the two most similar clusters and melt them until number of desired clusters is reached    
    while len(clusters) > limit:
        cluser0=0
        cluser1=1
        minimumCosine=10000

        #Find the two most similar clusters
        for i in range(0, len(clusters)):
            for j in range(0, len(clusters)):
                if j>i:
                    comparisons=0
                    sumCosine=0
                    newCluster=[]
                    newCluster=clusters[i]+clusters[j]
                    for elem in newCluster:
                        for elem2 in newCluster:
                            if elem!=elem2:
                                sumCosine+=distances[elem,elem2]
                    sim=sumCosine*(0.5)* (1/(len(newCluster)*(len(newCluster)-1)))        
                    if sim < minimumCosine:
                        minimumCosine=sim
                        cluser0=i
                        cluser1=j    
        #Melt the two found clusters 
        newClusters=[]
        newCluster=clusters[cluser0] + clusters[cluser1]
        del clusters[cluser1]
        del clusters[cluser0]
        clusters.append(newCluster)
    
    #Calculate the centroids
    centroids=[]
    vector0=np.zeros(shape=(1, len(vectors[0])))
    vector1=np.zeros(shape=(1, len(vectors[0])))
    for i in clusters[0]:
        vector0=np.add(vector0, vectors[i])
    for i in clusters[1]:
        vector1=np.add(vector1, vectors[i])
    return [vector0[0], vector1[0]]


def plotClusters(toCluster, gold, actual):
    embedding = MDS(n_components=2, metric=True, n_init=10, max_iter=500, random_state=100)
    X_transformed = embedding.fit_transform(toCluster)

    #Create list of different colors (Start with known colors)
    color = ["red", "blue", "green", "olive", "orange", "black", "lime", "deeppink"]
    for i in range(50):
        color.append('#%06X' % randint(0, 0xFFFFFF))

    #Plot the expected (gold) clustering
    plt.figure(figsize=(10, 20))
    plt.subplot(211)
    for k in set(gold):
        for i in range(0, len(toCluster)):  
            if gold[i]==k:
                plt.plot(X_transformed[i,0], X_transformed[i,1] , 'o',color=color[k], markersize=4)
    plt.title('Gold Labeling', fontsize=15, color='black')
    plt.xticks([])
    plt.yticks([])

    #Plot the actual clustering
    plt.subplot(212)
    for k in set(actual):
        for i in range(0, len(toCluster)):  
            if actual[i]==k:
                plt.plot(X_transformed[i,0], X_transformed[i,1] , 'o',color=color[k], markersize=4)
    plt.title('Actual labeling', fontsize=15, color='black')
    plt.xticks([])
    plt.yticks([])


if __name__ == '__main__':
    main()
