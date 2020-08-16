#!/usr/bin/env python
# coding: utf-8
# In[34]:

import warnings
warnings.filterwarnings("ignore")
from docopt import docopt
import logging
import time
import numpy as np
from utils_ import Space
import gzip
import math

from utils_ import Space
import count
import ppmi
import svd



def main():

    # Get the arguments
    args = docopt("""

    Usage:
        WordVectors.py  <representation> <pathCorpus> <outPathVectors> <outPathw2i>
        
    Arguments:
       
        <representation> = Either "COUNT", "PPMI" or "SVD"
        <pathCorpus> = Path to the corpus
        <outPathVectors> = Path for storing the vectors 
        <outPathw2i> = Path for storing w2i
    
    """)
    
    representation = args['<representation>']
    pathCorpus = args['<pathCorpus>']
    outPathVectors = args['<outPathVectors>']
    outPathw2i = args['<outPathw2i>']

    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    start_time = time.time()    
    
    #Create w2i
    logging.info("Create w2i")
    count=0
    with gzip.open(pathCorpus,'rt', encoding="utf-8") as sentences:
        setWV=set()
        listWV=[]
        try:
            for sentence in sentences:
                count+=1
                for word in sentence.split():
                    setWV.add(word)
        except:
            pass
        vocabulary=sorted(list(setWV))

    w2i = {w: i for i, w in enumerate(vocabulary) }       



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


    #Create co-occurence matrix
    logging.info("Create co-occurence matrix")             
    get_ipython().run_line_magic('run', 'count.py --len Data/ccoha2.txt.gz outPathVectors 20')
    
    if representation == "ppmi":
        #Apply PPMI
        logging.info("Apply PPMI")         
        get_ipython().run_line_magic('run', 'ppmi.py --len outPathVectors outPathVectors 1 1')
    
    if representation == "svd":
        #Apply PPMI
        logging.info("Apply PPMI") 
        get_ipython().run_line_magic('run', 'ppmi.py --len outPathVectors outPathVectors 1 1')
        #Apply SVD
        logging.info("Apply SVD") 
        get_ipython().run_line_magic('run', 'svd.py --len outPathVectors outPathVectors 100 0')
        

    #Save w2i
    logging.info("Save w2i")
    np.save(outPathw2i , w2i)


        
if __name__ == '__main__':
    main()


# In[33]:





# In[ ]:




