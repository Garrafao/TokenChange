# TokenChange

## Necessary files

Create 3 empty csv files and put it into the folder Storage/SecondOrder:

cluster.csv   --> In this file all cluster precision scores will be stored

lables.csv    --> In this file clustering labels will be stored 

lsc.csv       --> In this file all the semantic change score will be stored

## Necessary data (store it in the Data folder):

Corpus (In the example I used ccoha2)

Pretrained word2vec model

## Used packages

The following files are used from https://github.com/Garrafao/LSCDetection:

utils_.py

svd.py

ppmi.py 

count.py 

## Installations

To use the BERT methods of this repository, the torch package and the transformers package need to be installed, this can be done by using the following commands: 

pip install torch --user

pip install transformers --user

## Example Word sense clustering


The first set of methods is for applying word sense clustering of the uses of a specific word. All methods can be found in the WordSenseClustering folder: 


### Example count based: 

1) Create a vector for each type of a corpus by counting:
```python 
ipython WordSenseClustering/WordVectors.py count Data/ccoha2.txt.gz Storage/FirstOrder/Vectors.npz Storage/FirstOrder/w2i.npz.npy
```
2) Create token vectors of sample sentences by summing up all co-occurring type vectors
```python 
ipython WordSenseClustering/CountBasedVectors.py Storage/FirstOrder/Vectors.npz Data/monetary.csv Storage/FirstOrder/w2i.npz.npy Storage/SecondOrder/Vectors.npz 20 Data/ccoha2.txt.gz
```
3) Cluster the vectors and compare it to the epected clusterin 
```python 
ipython WordSenseClustering/Clustering.py Storage/SecondOrder/Vectors.npz Data/monetary.csv gaac 2 Storage/SecondOrder/lables.csv Storage/SecondOrder/cluster.csv

```


### Example word2vec: 

1) Create a vector for each type of a corpus by counting to get the iDf values: 
```python 
ipython WordSenseClustering/WordVectors.py count Data/ccoha2.txt.gz Storage/FirstOrder/Vectors.npz Storage/FirstOrder/w2i.npz.npy
```
2) Create token vectors of sample sentences by summing up all co-occurring type vectors, given by googles word2vec
```python 
ipython WordSenseClustering/W2v.py Data/monetary.csv Storage/FirstOrder/w2i.npz.npy Storage/SecondOrder/Vectors.npz 20 Data/ccoha2.txt.gz
```
3) Cluster the vectors and compare it to the epected clusterin
```python
ipython WordSenseClustering/Clustering.py Storage/SecondOrder/Vectors.npz Data/monetary.csv gaac 2 Storage/SecondOrder/lables.csv Storage/SecondOrder/cluster.csv

```



### Example Bert:

1) Create lemmatized token vectors of sample sentences using googles BERT
```python
ipython WordSenseClustering/Bert.py Data/monetary.csv Storage/SecondOrder/Vectors.npz lemma
```
2)Cluster the vectors and compare it to the epected clusterin 
```python
ipython WordSenseClustering/Clustering.py Storage/SecondOrder/Vectors.npz Data/monetary.csv gaac 2 Storage/SecondOrder/lables.csv Storage/SecondOrder/cluster.csv

```


## Example Lexical semantic change detection
The scripts creates the vectors for sentences of two times and clusters them and then automatically saves the semtantic change scores in a file. 

### Example count based:
### Example word2vec: 

### Example Bert:
```python
ipython SemanticChangeDetection/LSC_Bert.py Data/monetary.csv Data/monetary.csv Storage/SecondOrder/Vectors.npz Storage/SecondOrder/labels.csv lemma gaac 0 Storage/SecondOrder/lsc.csv 0.2 0.02 10


```












