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
import gzip

from utils_ import Space

import torch
from transformers import BertTokenizer, BertModel


def main():

    # Get the arguments
    args = docopt("""

    Usage:
        Bert.py  <pathTestSentences> <outPathVectors> <vecType> 
        
    Arguments:
       
        <pathTestSentences> = Path to the test sentences
        <outPathVectors> = Path for storing the vectors
        <vecType> = "token" or "lemma"

    """)
    
    pathTestSentences = args['<pathTestSentences>']
    outPathVectors = args['<outPathVectors>']
    vecType = args['<vecType>']
    

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.CRITICAL)
    print("")
    start_time = time.time()    
    logging.critical("Bert start")
 
    
    
    #Load TestSentences 
    # Load pre-trained model tokenizer (vocabulary)
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load pre-trained model (weights)
    global model
    model = BertModel.from_pretrained('bert-base-uncased',
                          output_hidden_states = True)


    contextVectorList=[]
    testSentences=[]
    with open(pathTestSentences, 'r') as csvFile:
        reader = csv.DictReader(csvFile, delimiter="\t")
        for row in reader:
            testSentences.append(dict(row))  
   
        #Token vs. Lemma
        if vecType == "token":
            vecTypeString = "sentence_token"
        else:
            vecTypeString = "sentence"

        #Create the vectors  
        logging.critical("Create Bert embeddings")         
        for i in range(0, len(testSentences)): 
            #Create target word(s)
            targetWord=str(testSentences[i][vecTypeString].split()[int([testSentences[i]["target_index"]][0])])
            targetWords=[]
            targetWords.append(tokenizer.tokenize(targetWord))
            targetWords=targetWords[0]
            
            #Tokenize text
            text=testSentences[i][vecTypeString]            
            marked_text = "[CLS] " + text + " [SEP]"
            tokenized_text = tokenizer.tokenize(marked_text)
            
            #Search the indices of the tokenized target word in the tokenized text
            targetWordIndices=[]
    
            for i in range(0, len(tokenized_text)):
                if tokenized_text[i] == targetWords[0]:
                    for l in range(0, len(targetWords)):
                        if tokenized_text[i+l] == targetWords[l]:
                            targetWordIndices.append(i+l)
                        if len(targetWordIndices) == len(targetWords):
                            break             

            #Create BERT Token Embeddings        
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            model.eval() 
            with torch.no_grad():
                outputs = model(tokens_tensor, segments_tensors)
                hidden_states = outputs[2]
            token_embeddings = torch.stack(hidden_states, dim=0)
            token_embeddings = torch.squeeze(token_embeddings, dim=1)
            token_embeddings = token_embeddings.permute(1,0,2)
            vectors=[]
            for number in targetWordIndices:
                token=token_embeddings[number]
                sum_vec=np.sum([np.array(token[12]),np.array(token[1])], axis=0)
                vectors.append(np.array(sum_vec))
            contextVectorList.append(np.sum(vectors, axis=0, dtype=float))
            
    #Normalize vectors in length
    contextVectorList=preprocessing.normalize(contextVectorList, norm='l2')

    #Save contextVectorList_sparse matrix
    outSpace = Space(matrix = contextVectorList, rows=" ", columns=" ")
    outSpace.save(outPathVectors)
      
    logging.critical("Bert end")  		
    logging.critical("--- %s seconds ---" % (time.time() - start_time))
    print("")

   

    

#Method that has as input a list of vectors and outputs the sum of the vectors
def getContextVector(toMeltList):
    centroid = lambda inp: [[sum(m) for m in zip(*l)] for l in inp]
    return centroid([toMeltList])[0] 
        
    

if __name__ == '__main__':
    main()

