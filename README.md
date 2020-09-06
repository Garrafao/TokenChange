# TokenChange

Repository containing code for [this Bachelor Thesis](#bibtex).

## Contents

### Word Sense Clustering
 The first part of my bachelor thesis deals with the automatic understanding of the uses of ambigous words. One way to understand the meaning of word uses is to create (token) vectors for each individual word use. Token vectors can be created in many different ways and in my work three different ones were compared (for references refer to my thesis): 

1. Self-trained count-based type vectors:
First learn count-based (count+PPMI+SVD) type vectors from a corpus then sum up all type vectors that co-occur with the word use, using the words inverse document frequency (iDf) as weight, since it improves the result.  

2. Pretrained type vectors from Google's word2vec:
First download pre-trained word2vec (SGNS) type vectors, then sum up all type vectors that co-occur with the word use, using the words inverse document frequency as weight, since it improves the result.  

3. Pretrained token vectors from BERT:
First download pre-trained BERT vectors, then calculate pretrained token vectors using BERT.

After the creation of token vectors they can be clustered into clusters of uses with similar meanings. Here this is done by initializing K-means clustering with precalculated centroids by applying Group-Average-Agglomerative-Clustering on a sample of vectors.

The performance of the clustering is measured by comparing the expected (human-annotated) clustering labels with the actual clustering labels using the Mean Adjusted Rand Index and Cluster Accuracy.

### Lexical Semantic Change Detection
 The second part of my bachelor thesis deals with the discovery of lexical semantic change. This is done by creating and comparing token vectors of two different times. 

Three different comparison measures are used for finding graded LSC values:

1. Average pairiwse distance (APD): 
Given two lists of token vectors (one for each period of time), where one vector represents one use of the word in this period. The APD chooses a sample of vectors from both times and measures their average cosine distance. A high average distance between the two times indicates a change in the usage of the word.

2. Cosine similarity (COS):
The idea is to average all the vectors from both periods of time and then compare these two average vectors by using the cosine distance.  A high cosine distance between the two average vectors indicates a change in the usage of the word.

3. Jensen Shannon Distance (JSD):
The third measure is more complex, a clustering of all the token vectors from both periods of time together needs to be performed. The resulting labels of the clustering can then be divided into the labels that correspond to the vectors from the first period of time and the vectors from the second period. Then the two lists of labels are compared using the Jensen-Shannon Distance, that compares the usage distributions of the two clusterings and returns a high value, if there is a change in the usage.

And three measures for finding binary LSC values:

1) APD: 
All words with a graded APD value above a certain threshold are assigned the value 1 and all below the threshhold the value 0.

2) COS: 
All words with a graded COS value above a certain threshold are assigned the value 1 and all below the threshhold the value 0.

3) Cluster based: 
The occurrences of the words from both corpora are clustered together and if there is a cluster, which contains more or equal k elements from the one time and less than k at the other time, the word is assigned the change value 1. else 0.


For more information check [this Bachelor Thesis](#bibtex).

The repository contains two different types of scripts: 

1. `WordSenseClustering/`: Contains python scripts for creation of token vectors and applying word sense clustering using different token vector representations
2. `SemanticChangeDetection/`: Contains python scripts for measuring the semantic change of words using different token vector representations. 

## Usage

All scripts should be run from the main directory. Note that all scripts expect several parameter values. What parameters these are and what values they expect can be seen from the scripts. Below are examples for all scripts and their parameters. All scripts can be run directly from the command line:   

	ipython WordSenseClustering/Bert.py <pathTestSentences> <outPathVectors> <vecType>

e.g.

	ipython WordSenseClustering/Bert.py Data/monetary.csv Files/Vectors/SecondOrder/Vectors.npz lemma

The usage of each script can be understood by running it with help option `-h`, e.g.:

	ipython WordSenseClustering/Bert.py -h

We recommend you to run the scripts within a [virtual environment](https://pypi.org/project/virtualenv/) with Python 3.8.5. 


## Data 

After executing the import.py script with
```python 
ipython import.py
```
the `Data/` folder will contain the lemmatized [CCOHA2](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/sem-eval-ulscd-eng/) corpus and a file that contains test sentences for the pseudoword "monetary/gothic". Please additionally download the [word2vec vectors](https://drive.google.com/uc?export=download&confirm=3aS5&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM) and move the .bin file into the `Data/` folder.

In the following examples I will only use the lemmatized corpus and lemmatized test sentences (The ccoha corpora are available both, lemmatized and non-lemmatized). It is also possible to use non-lemmatized test sentences, which performed better (see [this Bachelor Thesis](#bibtex))

Note that if you want to create self-trained, count-based token vectors for non-lemmatized sentences, the corpora on which the type vectors are trained on, has to contain non-lemmatized sentences too! The word2vec vocabulary only contains lemmatized words, so it can not be used on non-lemmatized test sentences.

BERT model can handle both, lemmatized and non-lemmatized test sentences.
, just by changing the parameter "lemma" to "token".

The test sentences should be stored in a csv file with the following values for each sentence: 

sentence: The lemmatized sentence. 
target_index: The index of the target word in the sentence. 
cluster: The expected cluster ID to which the word occurence belongs.
original_word: In case of lemmatazation or pseudowords it is necessary to know what the original word was. 
sentence_token: The non-lemmatized sentence
sentence_pos: For each word its part-of-speach.

## External packages

The following files are drawn from [LSCDetection](https://github.com/Garrafao/LSCDetection).

- utils_.py
- svd.py
- ppmi.py 
- count.py 

They will be downloaded, after running the import script: 
```python 
ipython import.py
```

After that install additional required packages running:
```python 
pip install -r requirements.txt 
```

## Example word sense clustering

The first set of methods is for creating token vectors and applying word sense clustering to the uses of a specific word. The clustering performance scores will automatically be stored (mean adjusted rand index and cluster accuracy) into a file (`Files/Clustering/cluster_scores.csv`). All methods can be found in the `WordSenseClustering/` folder.


### Example count-based: 

1) Create a type vector and the iDf value for each word type of the CCOHA2 corpus by counting and applying PPMI and SVD reduction.  
```python 
ipython WordSenseClustering/WordVectors.py svd Data/ccoha2.txt.gz Files/Vectors/FirstOrder/matrix.npz Files/Vectors/FirstOrder/w2i.npz.npy
```
2) Create token vectors for all occurences of a pseudoword ("monetary/gothic") by summing up all co-occurring type vectors, using their iDf value as weight.
```python 
ipython WordSenseClustering/CountBasedVectors.py Files/Vectors/FirstOrder/matrix.npz Data/monetary.csv Files/Vectors/FirstOrder/w2i.npz.npy Files/Vectors/SecondOrder/Vectors.npz 20 Data/ccoha2.txt.gz
```
3) Cluster the vectors and compare to expected clustering.
```python 
ipython WordSenseClustering/Clustering.py Files/Vectors/SecondOrder/Vectors.npz Data/monetary.csv gaac 2 Files/Clustering/cluster_labels.csv Files/Clustering/cluster_scores.csv

```


### Example word2vec: 

1) Create a vector and its iDf value for each type of the CCOHA2 corpus by counting. The iDf values are needed. 
```python 
ipython WordSenseClustering/WordVectors.py count Data/ccoha2.txt.gz Files/Vectors/FirstOrder/matrix.npz Files/Vectors/FirstOrder/w2i.npz.npy
```
2) Create token vectors of sample occurences of the pseudoword ("monetary/gothic") by summing up all co-occurring type vectors, given by Google's word2vec, using their iDf value as weight.
```python 
ipython WordSenseClustering/W2v.py Data/monetary.csv Files/Vectors/FirstOrder/w2i.npz.npy Files/Vectors/SecondOrder/Vectors.npz 20 Data/ccoha2.txt.gz
```
3) Cluster the vectors and compare to expected clustering.
```python
ipython WordSenseClustering/Clustering.py Files/Vectors/SecondOrder/Vectors.npz Data/monetary.csv gaac 2 Files/Clustering/cluster_labels.csv Files/Clustering/cluster_scores.csv

```



### Example BERT:

1) Create lemmatized token vectors of sample occurences of the pseudoword ("monetary/gothic") by using Google's BERT
```python
ipython WordSenseClustering/Bert.py Data/monetary.csv Files/Vectors/SecondOrder/Vectors.npz lemma
```
3) Cluster the vectors and compare to expected clustering.
```python
ipython WordSenseClustering/Clustering.py Files/Vectors/SecondOrder/Vectors.npz Data/monetary.csv gaac 2 Files/Clustering/cluster_labels.csv Files/Clustering/cluster_scores.csv

```


## Example lexical semantic change detection
The scripts create token vectors for sentences of two time periods and cluster them. Then they automatically save the semantic change scores (APD, COS, JSD) and the actual clustering labels in a file (`Files/LSC/lsc_scores.csv` and `Files/Clustering/cluster_labels.csv`). In this example both test sentences are identical, so the semantic change score should be close to 0.0. 

### Example word2vec:
```python
ipython SemanticChangeDetection/LSC_W2V.py Data/monetary.csv Data/monetary.csv Files/Vectors/SecondOrder/Vectors.npz Files/Clustering/cluster_labels.csv gaac Files/LSC/lsc_scores.csv 0.2 0.02 10 20  Files/Vectors/FirstOrder/w2i.npz.npy Data/ccoha2.txt.gz
```
### Example count-based: 
```python
ipython SemanticChangeDetection/LSC_SVD.py Data/monetary.csv Data/monetary.csv Files/Vectors/SecondOrder/Vectors.npz Files/Clustering/cluster_labels.csv gaac Files/LSC/lsc_scores.csv 0.2 0.02 10 Files/Vectors/FirstOrder/matrix.npz Files/Vectors/FirstOrder/w2i.npz.npy 20 Data/ccoha2.txt.gz
```
### Example BERT:
```python
ipython SemanticChangeDetection/LSC_Bert.py Data/monetary.csv Data/monetary.csv Files/Vectors/SecondOrder/Vectors.npz Files/Clustering/cluster_labels.csv lemma gaac Files/LSC/lsc_scores.csv 0.2 0.02 10
```




BibTex
--------

```
@bachelorsthesis{Laicher2020,
title={{Historical word sense clustering with deep contextualized word embeddings}},
author={Laicher, Severin},
year={2020},
school = {Institute for Natural Language Processing, University of Stuttgart},
address = {Stuttgart}
}
```


