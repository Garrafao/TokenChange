# TokenChange

Repository containing code for [this Bachelor Thesis](#bibtex).

## Contents

This repository contains several methods for:

1. Creating contextualized token vectors 
2. Clustering token vectors into senses
3. Analysing the sematic change of words/tokens 

Based on: 

1. Count based vectors 
2. Pretrained word vetors from Google's word2vec 
3. Token embeddings created by BERT

The repository conatins the folders: 

1. WordSenseClustering: Contains several python scripts for creating token vectors and apllying word sense clustering. 
2. SemanticChangeDetection: Contains several python scripts for measuring the sematnic change of words/tokens.

## Usage


The scripts should be run directly from the main directory. All scripts can be run directly from the command line:

	python WordSenseClustering/Bert.py <pathTestSentences> <outPathVectors> <vecType>

e.g.

	python WordSenseClustering/Bert.py Data/monetary.csv Storage/SecondOrder/Vectors.npz lemma

The usage of each script can be understood by running it with help option `-h`, e.g.:

	python WordSenseClustering/Bert.py -h

We recommend you to run the scripts within a [virtual environment](https://pypi.org/project/virtualenv/) with Python 3.8.5. 

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
pip install -r requirements.txt
```

## Example Word sense clustering


The first set of methods is for applying word sense clustering to the uses of a specific word. The clustering performance will automatically be stored into a file. All methods can be found in the WordSenseClustering folder.


### Example count based: 

1) Create a vector for each word type in a corpus by counting:
```python 
python WordSenseClustering/WordVectors.py count Data/ccoha2.txt.gz Storage/FirstOrder/Vectors.npz Storage/FirstOrder/w2i.npz.npy
```
2) Create token vectors of sample sentences by summing up all co-occurring type vectors
```python 
python WordSenseClustering/CountBasedVectors.py Storage/FirstOrder/Vectors.npz Data/monetary.csv Storage/FirstOrder/w2i.npz.npy Storage/SecondOrder/Vectors.npz 20 Data/ccoha2.txt.gz
```
3) Cluster the vectors and compare to gold clustering 
```python 
python WordSenseClustering/Clustering.py Storage/SecondOrder/Vectors.npz Data/monetary.csv gaac 2 Storage/SecondOrder/lables.csv Storage/SecondOrder/cluster.csv

```


### Example word2vec: 

1) Create a vector for each type of a corpus by counting to get the iDf values: 
```python 
python WordSenseClustering/WordVectors.py count Data/ccoha2.txt.gz Storage/FirstOrder/Vectors.npz Storage/FirstOrder/w2i.npz.npy
```
2) Create token vectors of sample sentences by summing up all co-occurring type vectors, given by Google's word2vec
```python 
python WordSenseClustering/W2v.py Data/monetary.csv Storage/FirstOrder/w2i.npz.npy Storage/SecondOrder/Vectors.npz 20 Data/ccoha2.txt.gz
```
3) Cluster the vectors and compare to gold clustering
```python
python WordSenseClustering/Clustering.py Storage/SecondOrder/Vectors.npz Data/monetary.csv gaac 2 Storage/SecondOrder/lables.csv Storage/SecondOrder/cluster.csv

```



### Example BERT:

1) Create lemmatized token vectors of sample sentences using Google's BERT
```python
python WordSenseClustering/Bert.py Data/monetary.csv Storage/SecondOrder/Vectors.npz lemma
```
2) Cluster the vectors and compare to gold clustering 
```python
python WordSenseClustering/Clustering.py Storage/SecondOrder/Vectors.npz Data/monetary.csv gaac 2 Storage/SecondOrder/lables.csv Storage/SecondOrder/cluster.csv

```


## Example Lexical semantic change detection
The scripts create token vectors for sentences of two times and clusters them. Then it automatically saves the semantic change scores in a file. 

### Example count based:
```python
python SemanticChangeDetection/LSC_W2V.py Data/monetary.csv Data/monetary.csv Storage/SecondOrder/Vectors.npz Storage/SecondOrder/lables.csv gaac Storage/SecondOrder/lsc.csv 0.2 0.02 10 20 Storage/FirstOrder/w2i.npz.npy Data/ccoha2.txt.gz
```
### Example word2vec: 
```python
python SemanticChangeDetection/LSC_SVD.py Data/monetary.csv Data/monetary.csv Storage/SecondOrder/Vectors.npz Storage/SecondOrder/lables.csv gaac Storage/SecondOrder/lsc.csv 0.2 0.02 10 Storage/FirstOrder/Vectors.npz Storage/FirstOrder/w2i.npz.npy 20 Data/ccoha2.txt.gz
```
### Example BERT:
```python
python SemanticChangeDetection/LSC_Bert.py Data/monetary.csv Data/monetary.csv Storage/SecondOrder/Vectors.npz Storage/SecondOrder/lables.csv lemma gaac Storage/SecondOrder/lsc.csv 0.2 0.02 10
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


