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
        W2v.py  <pathTestSentences> <outPathVectors> <windowSize2> 
        
    Arguments:
       
        <pathTestSentences> = Path to the test sentences
        <outPathVectors> = Path for storing the vectors 
        <windowSize2> = Window size (20 works good)
    
    """)
    
  
    pathTestSentences = args['<pathTestSentences>']
    outPathVectors = args['<outPathVectors>']
    windowSize2 = int(args['<windowSize2>'])

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.CRITICAL)
    print("")
    start_time = time.time()    
    logging.critical("W2V start")
 
    
    if not isinstance(windowSize2, int):
        windowSize2 = 20
 
    #Load Word2Vec
    model = gensim.models.KeyedVectors.load_word2vec_format('Data/GoogleNews-vectors-negative300.bin', binary=True)  

    #Load w2i
    w2i = np.load(pathw2i, allow_pickle='TRUE').item()

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
    #self.target=str(self.testSentences[0]["original_word"])        
    for dic in self.testSentences:
        self.sentence = dic['sentence_token'].split()
        for i, word in enumerate(self.sentence):  
            if str(i) == dic['target_index']:

                self.toMelt=[]
                toMeltIDF=[]
                self.lowerWindowSize = max(i-self.windowSize, 0)
                self.upperWindowSize = min(i+self.windowSize, len(self.sentence))
                self.window = self.sentence[self.lowerWindowSize:i] + self.sentence[i+1:self.upperWindowSize+1] 
                if word in model.wv.vocab:
                    for contextWord in self.window:
                        if contextWord in model.wv.vocab:
                            if contextWord != "$":
                                self.toMelt.append(preprocessing.normalize([model.wv[contextWord]], norm='l2')[0]) 


                    self.contextVectorList.append(self.getContextVector(self.toMelt))  
                else:
                    self.contextVectorList.append(np.zeros(300))


    



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

    

