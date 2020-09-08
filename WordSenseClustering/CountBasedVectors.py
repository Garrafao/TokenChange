#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")
from docopt import docopt
import logging
import time
from scipy.cluster.vq import kmeans2
import numpy as np
import csv
from sklearn import preprocessing
import gzip
from scipy.spatial import distance
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn import metrics
import math 
from utils_ import Space
import sys

def main():

    # Get the arguments
    args = docopt("""

    Usage:
        CountBasedVectors.py  <pathMatrix> <pathw2i> <pathCorpus> <pathTestSentences> <outPathVectors> <sentenceType> <windowSize2> 
        CountBasedVectors.py  <pathCorpus> <pathTestSentences> <sentenceType> <windowSize2>
        
    Arguments:
       
        <pathMatrix> = Path to the word vector matrix
        <pathw2i> = Path to the word-to-index
        <pathCorpus> = path to the corpus 
        <pathTestSentences> = Path to the test sentences
        <outPathVectors> = Path for storing the vectors
        <sentenceType> = "lemma" or "token"
        <windowSize2> = Window size (20 works fine)
        
        
    """)
    
    pathMatrix = args['<pathMatrix>']
    pathTestSentences = args['<pathTestSentences>']
    pathw2i = args['<pathw2i>']
    outPathVectors = args['<outPathVectors>']
    windowSize2 = int(args['<windowSize2>'])
    pathCorpus = args['<pathCorpus>']
    sentenceType = args['<sentenceType>']

    
    if len(sys.argv) == 5:
        pathMatrix = "Files/Vectors/FirstOrder/matrix.npz"
        pathw2i = "Files/Vectors/FirstOrder/w2i.npz.npy"
        outPathVectors = "Files/Vectors/SecondOrder/Vectors.npz"
        

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.CRITICAL)
    print("")
    start_time = time.time()    
    logging.critical("ContextVectors start")
    
    #Load w2i
    w2i = np.load(pathw2i, allow_pickle='TRUE').item()

    if sentenceType == "token":
        sentType = "sentence_token"
    else:
        sentType = "sentence"
        

    #Load saved wordVectorMatrix
    try:
        inSpace =  Space(path=pathMatrix, format='w2v')
    except UnicodeDecodeError:
        inSpace = Space(path=pathMatrix)

        
    
    #inSpace =  Space(path=pathMatrix, format='w2v')
    #inSpace = Space(path=pathMatrix)
    cooc_mat_sparse=inSpace.matrix    
      
    #Calculate IDF for every word 
    docFreq={}
         
    for i in range(0, len(w2i)):
        docFreq[i]=0
    with gzip.open(pathCorpus,'rt', encoding="utf-8") as sentences: 
        count=0
        try:
            for sentence in sentences:
                count=count+1
                for word in set(sentence.split()):
                    if word in w2i:
                        docFreq[w2i[word]]+=1
        except:
            pass
        for key, value in w2i.items(): 
            docFreq[value] = math.log10(count/max(docFreq[value],1)) 

    #Load TestSentences 
    contextVectorList=[]
    testSentences=[]
    with open(pathTestSentences, 'r') as csvFile:
        reader = csv.DictReader(csvFile, delimiter="\t")
        for row in reader:
            testSentences.append(dict(row))   

    #Calculate contextVectorMatrix
    logging.critical("Calculate contextVectorMatrix")
    nonExisting=False
    target=str(testSentences[0]["original_word"])        
    for dic in testSentences:
        sentence = dic[sentType].split()
        for i, word in enumerate(sentence):  
            if str(i) == dic['target_index'] and word == target:
                toMelt=[]
                toMeltIDF=[]
                lowerWindowSize = max(i-windowSize2, 0)
                upperWindowSize = min(i+windowSize2, len(sentence))
                window = sentence[lowerWindowSize:i] + sentence[i+1:upperWindowSize+1] 
                if word in w2i:
                    windex = w2i[word]
                    for contextWord in window:
                        if contextWord != "$":
                            if contextWord in w2i:
                                contextWordIndex = w2i[contextWord]
                                toMelt.append(cooc_mat_sparse[contextWordIndex].toarray()[0]
                                             *math.pow(docFreq[contextWordIndex],1 ))  
                    contextVectorList.append(getContextVector(toMelt))  
                else:
                    nonExisting=True
  

    #Normalize vectors in length
    contextVectorList=preprocessing.normalize(contextVectorList, norm='l2')

    #Save contextVectorList_sparse matrix
    outSpace = Space(matrix = contextVectorList, rows=" ", columns=" ")
    outSpace.save(outPathVectors)
      
    logging.critical("ContextVectors end")  		
    logging.critical("--- %s seconds ---" % (time.time() - start_time))
    print("")
   
    
#Method that has as input a list of vectors and outputs the sum of the vectors
def getContextVector(toMeltList):
    centroid = lambda inp: [[sum(m) for m in zip(*l)] for l in inp]
    return centroid([toMeltList])[0] 
        
    
if __name__ == '__main__':
    main()
