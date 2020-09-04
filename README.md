# TokenChange

Repository containing code for [this Bachelor Thesis](#bibtex).

## Contents

The first part of my bachelor thesis deals with the automatic understanding of the uses of ambigous words. One way to understand the meaning of word uses is to create token based-vectors for each individual word use. Token vectors can be created in many different ways and in my work three different ones were presented: 

1. Count based vectors:
By summing up all self-trained, count-based, PPMI & SVD reduced word vectors that co-occur with the word use.

2. Pretrained word vetors from Google's word2vec 
By summing up all pretrained predictive word2vec word vectors that co-occur with the word use.

3. Token embeddings created by BERT:
By creating predictive token vectors using BERT.

After the creation of token vectors they can be clustered into uses with similar meanings. Here this is done by initializing K-means with precalculated centroids by applying Group-Average-Agglomerative-Clustering on a sample of vectors, in order to find a better solution.

The performance of the clustering is measured by comparing the expected clustering labels with the actual clustering labels using the adjusted rand index and the cluster accuracy measure.

The second part of my bachelor thesis deals with the discovery of lexical semantic change. This is done by creating and comparing token vectors of two different times. 

Three different comparison measures are used:

1. Average pairiwse distance (APD): 
Given two lists of token vectors (one for each period of time), where one vector represents one use of the word in this period. The APD chooses a sample of vectors from both times and measures their average mutual cosine distance. A high average distance between the two times indicates a change in the usage of the word.

2. Cosine similarity (COS):
The idea is to average all the vectors from both periods of time and then compare these two average vectors by using the cosine distance.

3. Jensen Shannon difference (JSD):
The third measure is more complex, a clustering of all the vectors from both periods of time together needs to be performed. The resulting labels of the clustering can then be divided into the labels that correspond to the vectors from the first period of time and the vectors from the second period. Then the two list of labels are compared by using  the Jensen-Shannon difference,  that compares the usage distributions of the two clusterings and returns a high value, if there is a change in the usage.

For more context information check [this Bachelor Thesis](#bibtex).

The repository conatins  two different types of methods: 

1. WordSenseClustering: Contains several python scripts for creating token vectors and applying word sense clustering using different token vector representations
2. SemanticChangeDetection: Contains several python scripts for measuring the semantic change of words using different token vector representations. 

## Usage

Not that all skripts expect several parameter values. What parameters these are and what values they expect can be taken from the scripts. Below are examples for all scripts and their parameters. All scripts can be run directly from the command line:   

	ipython WordSenseClustering/Bert.py <pathTestSentences> <outPathVectors> <vecType>

e.g.

	ipython WordSenseClustering/Bert.py Data/monetary.csv Files/Vectors/SecondOrder/Vectors.npz lemma

The usage of each script can be understood by running it with help option `-h`, e.g.:

	ipython WordSenseClustering/Bert.py -h

We recommend you to run the scripts within a [virtual environment](https://pypi.org/project/virtualenv/) with Python 3.8.5. 


## Data 

After executing the import.py script the Data folder will contain the lemmatized ccoha2 corpus (https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/sem-eval-ulscd-eng/), a file that contains test sentences for the pseudoword "monetary/gothic" and the pre trained word vectors from word2vec. (From where?)

In order to use the word2vec vectors, please dowload the vectors https://drive.google.com/uc?export=download&confirm=3aS5&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM and move the file into the Data folder.
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


After that install the required packages running:
```python 
pip install -r requirements.txt --user
```

## Example Word sense clustering


The first set of methods is for creating token vectors and applying word sense clustering to the uses of a specific word. The clustering performance scores will automatically be stored (adjusted rand index and cluster accuracy) into a file (`Files/Clustering/cluster_scores.csv`). All methods can be found in the WordSenseClustering folder.


### Example count based: 

1) Create a vector for each word type of the ccoha2 corpus by counting and applying PPMI and SVD reduction. 
```python 
ipython WordSenseClustering/WordVectors.py ppmi Data/ccoha2.txt.gz Files/Vectors/FirstOrder/matrix.npz Files/Vectors/FirstOrder/w2i.npz.npy
```
2) Create token vectors of sample occurences of the pseudoword ("monetary/gothic") by summing up all co-occurring type vectors.
```python 
ipython WordSenseClustering/CountBasedVectors.py Files/Vectors/FirstOrder/matrix.npz Data/monetary.csv Files/Vectors/FirstOrder/w2i.npz.npy Files/Vectors/SecondOrder/Vectors.npz 20 Data/ccoha2.txt.gz
```
3) Cluster the vectors and compare to expected clustering.
```python 
ipython WordSenseClustering/Clustering.py Files/Vectors/SecondOrder/Vectors.npz Data/monetary.csv gaac 2 Files/Clustering/cluster_labels.csv Files/Clustering/cluster_scores.csv

```


### Example word2vec: 

1) Create a vector for each type of the ccoha2 corpus by counting to get the inverse document values of each word.
```python 
ipython WordSenseClustering/WordVectors.py count Data/ccoha2.txt.gz Files/Vectors/FirstOrder/matrix.npz Files/Vectors/FirstOrder/w2i.npz.npy
```
2) Create token vectors of sample occurences of the pseudoword ("monetary/gothic") by summing up all co-occurring type vectors, given by Google's word2vec.
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


## Example Lexical semantic change detection
The scripts create token vectors for sentences of two times and clusters them. Then it automatically saves the semantic change scores (APD, COS, JSD) and the actual clustering labels in a file (`Files/LSC/lsc_scores.csv` and `Files/Clustering/cluster_labels.csv`). In this example both test sentences are identical, so the semantic change score should be zero or close tp zer0o. 

### Example count based:
```python
ipython SemanticChangeDetection/LSC_W2V.py Data/monetary.csv Data/monetary.csv Files/Vectors/SecondOrder/Vectors.npz Files/Clustering/cluster_labels.csv gaac Files/LSC/lsc_scores.csv 0.2 0.02 10 20 Files/Vectors/FirstOrder/matrix.npz Files/Vectors/FirstOrder/w2i.npz.npy Data/ccoha2.txt.gz
```
### Example word2vec: 
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


