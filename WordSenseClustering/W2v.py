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
import sys
import gensim




def main():

    # Get the arguments
    args = docopt("""

    Usage:
        W2v.py  <pathTestSentences> <outPathVectors> <windowSize2> <sentenceType>
        W2v.py  <pathTestSentences> <windowSize2> <sentenceType>
        
    Arguments:
       
        <pathTestSentences> = Path to the test sentences
        <outPathVectors> = Path for storing the vectors 
        <windowSize2> = Window size (20 works good)
        <sentenceType> = "lemma" or "token"
    
    """)
    
  
    pathTestSentences = args['<pathTestSentences>']
    outPathVectors = args['<outPathVectors>']
    windowSize2 = int(args['<windowSize2>'])
    sentenceType = args['<sentenceType>']

    if len(sys.argv) == 4:
        outPathVectors = "Files/Vectors/SecondOrder/Vectors.npz"
        
 
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.CRITICAL)
    print("")
    start_time = time.time()    
    logging.critical("W2V start")
 
    if sentenceType == "token":
        sentType = "sentence_token"
    else:
        sentType = "sentence"
        
    
    if not isinstance(windowSize2, int):
        windowSize2 = 20
 
    #Load Word2Vec
    model = gensim.models.KeyedVectors.load_word2vec_format('Data/GoogleNews-vectors-negative300.bin', binary=True)  

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
    #self.target=str(testSentences[0]["original_word"])        
    for dic in testSentences:
        sentence = dic[sentType].split()
        for i, word in enumerate(sentence):  
            if str(i) == dic['target_index']:

                toMelt=[]
                toMeltIDF=[]
                lowerWindowSize = max(i-windowSize2, 0)
                upperWindowSize = min(i+windowSize2, len(sentence))
                window = sentence[lowerWindowSize:i] + sentence[i+1:upperWindowSize+1] 
                if word in model.wv.vocab:
                    for contextWord in window:
                        if contextWord in model.wv.vocab:
                            if contextWord != "$":
                                toMelt.append(preprocessing.normalize([model.wv[contextWord]], norm='l2')[0]) 


                    contextVectorList.append(getContextVector(toMelt))  
                else:
                    contextVectorList.append(np.zeros(300))


    



    #Normalize vectors in length
    contextVectorList=preprocessing.normalize(contextVectorList, norm='l2')

    #Save contextVectorList_sparse matrix
    outSpace = Space(matrix = contextVectorList, rows=" ", columns=" ")
    outSpace.save(outPathVectors)
      
    logging.critical("W2V end")  		
    logging.critical("--- %s seconds ---" % (time.time() - start_time))
    print("")
   

    

#Method that has as input a list of vectors and outputs the sum of the vectors
def getContextVector(toMeltList):
    centroid = lambda inp: [[sum(m) for m in zip(*l)] for l in inp]
    return centroid([toMeltList])[0] 
        
    

if __name__ == '__main__':
    main()

    

