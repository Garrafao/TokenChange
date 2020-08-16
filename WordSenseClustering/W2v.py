#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import warnings
warnings.filterwarnings("ignore") 
from docopt import docopt
import logging
import time
import numpy as np
import csv
from sklearn import preprocessing
from utils_ import Space
import gzip
import gzip
import math
from utils_ import Space

import gensim




def main():

    # Get the arguments
    args = docopt("""

    Usage:
        W2v.py  <pathTestSentences> <pathw2i> <outPathVectors> <windowSize2> <pathCorpus>
        
    Arguments:
       
        <pathTestSentences> = Path to the test sentences
        <pathw2i> = <pathw2i>
        <outPathVectors> = Path for storing the vectors 
        <windowSize2> = Window size
        <pathCorpus> = path to the corpus 
    
    """)
    
  
    pathTestSentences = args['<pathTestSentences>']
    pathw2i = args['<pathw2i>']
    outPathVectors = args['<outPathVectors>']
    windowSize2 = int(args['<windowSize2>'])
    pathCorpus = args['<pathCorpus>']

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    start_time = time.time()    
    
    #Load Word2Vec
    logging.info("Load Word2Vec")
    model = gensim.models.KeyedVectors.load_word2vec_format('Data/GoogleNews-vectors-negative300.bin', binary=True)  

    #Load w2i
    logging.info("Load w2i")
    w2i = np.load(pathw2i, allow_pickle='TRUE').item()
      
    #Calculate IDF for every word 
    docFreq={}
    logging.info("Calculate IDF for every word ")
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
    logging.info("Load TestSentences and calculate contextVectorMatrix")
    contextVectorList=[]
    testSentences=[]
    with open(pathTestSentences, 'r') as csvFile:
        reader = csv.DictReader(csvFile, delimiter="\t")
        for row in reader:
            testSentences.append(dict(row))  
    
    #Calculate contextVectorMatrix
    logging.info("Calculate contextVectorMatrix") 
    nonExisting=False
    target=str(testSentences[0]["original_word"])        
    for dic in testSentences:
        sentence = dic['sentence'].split()
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
                        if contextWord in w2i:
                            if contextWord in model.wv.vocab:
                                if contextWord != "$":
                                    contextWordIndex = w2i[contextWord]
                                    toMelt.append(preprocessing.normalize([model.wv[contextWord]], norm='l2')[0]
                                                  *math.pow(docFreq[contextWordIndex],1 ))  
                    contextVectorList.append(getContextVector(toMelt))  
                else:
                    nonExisting=True

    if nonExisting:
        print("WORD UNKNOWN")

    #Normalize vectors in length
    logging.info("Normalize vectors in length")
    contextVectorList=preprocessing.normalize(contextVectorList, norm='l2')

    #Save contextVectorList_sparse matrix
    logging.info("Save contextVectorList_sparse matrix")
    outSpace = Space(matrix = contextVectorList, rows=" ", columns=" ")
    outSpace.save(outPathVectors)
      
    logging.info("--- %s seconds ---" % (time.time() - start_time))
   

    

#Method that has as input a list of vectors and outputs the sum of the vectors
def getContextVector(toMeltList):
    centroid = lambda inp: [[sum(m) for m in zip(*l)] for l in inp]
    return centroid([toMeltList])[0] 
        
    

if __name__ == '__main__':
    main()

    

