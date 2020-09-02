#!/usr/bin/env python
# coding: utf-8
# In[ ]:

from scipy.spatial import distance
import warnings
warnings.filterwarnings("ignore") 
from docopt import docopt
import logging
import time
import numpy as np
import csv
from sklearn import preprocessing
import gzip
import os 
from utils_ import Space




def main():

    # Get the arguments
    args = docopt("""

    Usage:
        LSC_W2V.py  <pathSentences1> <pathSentences2> <outPathVectors> <outPathLabels> <clusteringInitialization> <pathResults> <limitAGL> <limitCOS> <limitCluster> <windowSize> <pathToW2i> <pathCorpora>
        
    Arguments:
       
        <pathSentences1> = Path to the test sentences from time1
        <pathSentences2> = Path to the test sentences from time2
        <outPathVectors> = Path to store the vectors
        <outPathLabels> = Path to store the clustering labels
        <clusteringInitialization> = "gaac" for precalculated initializations, else random
        <pathResults> = Path to store the lsc scores
        <limitAGL> = Change score limit for AGL to still be consiered as change (Good is about 0.2)
        <limitCOS> = Change score limit for Cosine to still be consiered as change (Good is about 0.02) 
        <limitCluster> = Minimum size of a cluster to be a cluster (Good is 5-10)
        <windowSize> = Window size for words to be in context of other words (Good is 20)
        <pathToW2i> = Path to W2i
        <pathCorpora> = Path to the corpora


    """)
    
    pathSentences1 = args['<pathSentences1>']
    pathSentences2 = args['<pathSentences2>']
    outPathVectors = args['<outPathVectors>']
    outPathLabels = args['<outPathLabels>']
    clusteringInitialization = args['<clusteringInitialization>']
    pathResults = args['<pathResults>']
    limitAGL = float(args['<limitAGL>'])
    limitCOS = float(args['<limitCOS>'])
    limitCluster = int(args['<limitCluster>'])
    windowSize = int(args['<windowSize>'])
    pathToW2i = args['<pathToW2i>']
    pathCorpora = args['<pathCorpora>']


   

    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.CRITICAL)
    print("")
    start_time = time.time()    
    logging.critical("W2v LSC start")
    
    #Create and cluster the vectors of corpora 1
    logging.critical("Create and cluster the vectors of corpora 1")
    get_ipython().run_line_magic('run', 'WordSenseClustering/W2v.py $pathSentences1 $pathToW2i $outPathVectors $windowSize $pathCorpora')
    inSpace = Space(path=outPathVectors)
    vectors1=inSpace.matrix.toarray()
    get_ipython().run_line_magic('run', 'WordSenseClustering/Clustering.py $outPathVectors 0 $clusteringInitialization 0 $outPathLabels 0')
    labels1=[]
    with open(outPathLabels , 'r') as file:
        data = file.readlines()  
    for i in data[-1]:
        if i != ",":
            if i != "\n":
                labels1.append(int(i))    
   
    #Create and cluster the vectors of corpora 2
    logging.critical("Create and cluster the vectors of corpora 2")    
    get_ipython().run_line_magic('run', 'WordSenseClustering/W2v.py $pathSentences2 $pathToW2i $outPathVectors $windowSize $pathCorpora')
    inSpace = Space(path=outPathVectors)
    vectors2=inSpace.matrix.toarray()   
    get_ipython().run_line_magic('run', 'WordSenseClustering/Clustering.py $outPathVectors 0 $clusteringInitialization 0 $outPathLabels 0')
    labels2=[]
    with open(outPathLabels , 'r') as file:
        data = file.readlines() 
    for i in data[-1]:
        if i != ",":
            if i != "\n":
                labels2.append(int(i))         
    
    #Create the lists to store the binary results in 
    cosineDistanceBinary=[]
    APDBinary=[]
    clusterScoreBinary=[]

    #Calculate cosineDistance for the two vectors
    cosineDistance = getCOS(vectors1, vectors2)
    if cosineDistance >= limitCOS:
        cosineDistanceBinary.append(1)
    else:
        cosineDistanceBinary.append(0) 

    #Calculate Average pairwise distance for the two vectors
    APD = getAPD(vectors1, vectors2, 200)
    if APD >= limitAGL:
        APDBinary.append(1)
    else:
        APDBinary.append(0) 
 
    #Create and cluster the combined vectors of both corpora
    logging.critical("Create and cluster the combined vectors of both corpora")
    vectors = np.concatenate((vectors1, vectors2), axis=0)
    outSpace = Space(matrix = vectors, rows=" ", columns=" ")
    outSpace.save(outPathVectors)
    #Cluster the combined vectors
    get_ipython().run_line_magic('run', 'WordSenseClustering/Clustering.py $outPathVectors 0 $clusteringInitialization 0 $outPathLabels 0')
    
    
    #Load list of labels
    labels=[]
    with open(outPathLabels , 'r') as file:
        data = file.readlines()  
    for i in data[-1]:
        if i != ",":
            if i != "\n":
                labels.append(int(i))   

    # Calculated cluster LSC score
    labelA_1 = []
    labelA_2 = []

    maximum = len(vectors1)
    for i in range(0, len(vectors1)):
        labelA_1.append(labels[i])

    for i in range(maximum, maximum + len(vectors2)):
        labelA_2.append(labels[i])

    changeA=0
    for j in set(labels):
        if labelA_1.count(j) >= limitCluster :
            if labelA_2.count(j) < limitCluster :
                changeA=1
        if labelA_2.count(j) >= limitCluster :
            if labelA_1.count(j) < limitCluster:
                changeA=1
                
    clusterScoreBinary.append(changeA)

    p = np.histogram(labelA_1)[0] / len(labelA_1) 
    q = np.histogram(labelA_2)[0] / len(labelA_2) 
 
    dist = distance.jensenshannon(p, q)
    
    p = np.histogram(labels1)[0] / len(labels1)
    q = np.histogram(labels2)[0] / len(labels2)
    clusterScore = distance.jensenshannon(sorted(p),sorted(q))
    
    print("cosine distance:")
    print(cosineDistance)
    print("")
    print("Average pairwise distance:")
    print(APD)
    print("")
    print("JSD:")
    print(dist)
    print("")
  
    filename1 = os.path.splitext(os.path.basename(pathSentences1))[0]
    filename2 = os.path.splitext(os.path.basename(pathSentences2))[0]

    cos=[filename1, filename2, "cosineDistance",cosineDistance]
    apd=[filename1, filename2, "APD",APD]
    cluster=[filename1, filename2, "clusterScore", dist]
    cosBin=[filename1, filename2, "cosineDistanceBinary",cosineDistanceBinary[0]]
    APDBin=[filename1, filename2, "APDBinary",APDBinary[0]]
    clusterBin=[filename1, filename2, "clusterScoreBinary",clusterScoreBinary[0]]

    with open(pathResults, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([cos, apd, cluster, cosBin, APDBin, clusterBin])    
    
    logging.critical("W2v LSC end")  		
    logging.critical("--- %s seconds ---" % (time.time() - start_time))
    print("")

    
#Method that calcualtes the cosine distance of the two times    
def getCOS(vec1, vec2):
    sum1=np.sum(vec1, axis=0)/len(vec1)
    sum2=np.sum(vec2, axis=0)/len(vec2)
    result= distance.cosine(sum1, sum2)
    return result
    
    
#Method that calcualtes the average pairwise distance of the two times   
def getAPD(vec1, vec2, size):

    testList1=[]
    testList2=[]
    result=0
    size = min(len(vec1), len(vec2), size)
    randoms=random.sample(range(0, min(len(vec1), len(vec2))), size)
    for i in randoms: 
        testList1.append(vec1[i]) 
        testList2.append(vec2[i]) 

    for i in range(0, size):
        for j in range(0, size):
            result=result + distance.cosine(testList1[i], testList2[j])
    if result!=0:
        result=result/(size*size)

    return result

#Method that calcualtes the jensen shannon distance of the two times   
def jensen_shannon_distance(p, q):
    """
    method to compute the Jenson-Shannon Distance 
    between two probability distributions
    """

    # convert the vectors into numpy arrays in case that they aren't
    p = np.array(p)
    q = np.array(q)

    # calculate m
    m = (p + q) / 2

    # compute Jensen Shannon Divergence
    divergence = (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)) / 2

    # compute the Jensen Shannon Distance
    distance = np.sqrt(divergence)

    return distance


    
    
    
if __name__ == '__main__':
    main()

