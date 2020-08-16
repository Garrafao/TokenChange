#!/usr/bin/env python
# coding: utf-8

# In[5]:


#!/usr/bin/env python
# coding: utf-8

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
from random import randint
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

def main():

    # Get the arguments
    args = docopt("""

    Usage:
        count.py  <pathVectors> <pathTestSentences> <initializationType> <numberClusters> 
        
    Arguments:
       
        <pathVectors> = Path to the vectors
        <pathTestSentences> = Path to the test sentecens that contain the gold clustering
        <initializationType> = "gaac" for precalculated initialization, else random
        <numberClusters> = Number of desired clusters, if 0 than its calculated by sillhouette
    
    """)
    
    pathVectors = args['<pathVectors>']
    pathTestSentences = args['<pathTestSentences>']
    initializationType = args['<initializationType>']
    numberClusters = int(args['<numberClusters>'])
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    start_time = time.time()    
    
    #Load vectors
    logging.info("Load vectors") 
    inSpace = Space(path=pathVectors)
    loaded_contextVectorList_sparse=inSpace.matrix

    #Get gold clustering
    logging.info("Get gold clustering") 
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
    else:
        maxIndex=numberClusters
      
    
    if initializationType == "gaac":

        #Calculate GAAC on sample vectors for initial centroids
        logging.info("Calculate GAAC on sample vectors for initial centroids")
        testList=[]
        size = min(len(loaded_contextVectorList_sparse.toarray()), 50 )
        randoms=random.sample(range(0, len(loaded_contextVectorList_sparse.toarray())), size)
        for i in randoms: 
            testList.append(loaded_contextVectorList_sparse[i].toarray()[0])   
        initialCentroids=preprocessing.normalize(gaac(testList, maxIndex), norm='l2')

        #Calculate kmeans 
        logging.info("Calculate kmeans")    
        centroid, label = kmeans2(loaded_contextVectorList_sparse.toarray(),
                                                        initialCentroids , 5, minit='matrix')

    else:
        #Calculate kmeans 
        logging.info("Calculate kmeans")    
        centroid, label = kmeans2(loaded_contextVectorList_sparse.toarray(),
                                                        maxIndex , 5, minit='points')
        
        
    #Show results 
    logging.info("Show results")
    print("Adjusted rand index:")
    print(adjusted_rand_score(gold, label))
    print("Accuracy:")
    print(cluster_accuracy(np.array(gold), np.array(label)))
    plotClusters(loaded_contextVectorList_sparse.toarray(), gold, label)                                  

    logging.info("--- %s seconds ---" % (time.time() - start_time))
    
    
    
    
    
    
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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




