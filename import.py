#!/usr/bin/env python
# coding: utf-8
# In[34]:
import os

os.system("wget https://github.com/Garrafao/LSCDetection/archive/master.zip")
os.system("unzip master.zip")
os.system("rm master.zip")

os.system("mv LSCDetection-master/representations/count.py count.py")
os.system("mv LSCDetection-master/representations/svd.py svd.py")
os.system("mv LSCDetection-master/representations/ppmi.py ppmi.py")
os.system("mv LSCDetection-master/modules/utils_.py utils_.py")




os.system("wget https://www2.ims.uni-stuttgart.de/data/sem-eval-ulscd/semeval2020_ulscd_eng.zip")

os.system("unzip semeval2020_ulscd_eng.zip")
os.system("rm semeval2020_ulscd_eng.zip")


os.system("mv semeval2020_ulscd_eng/corpus2/lemma/ccoha2.txt.gz Data/ccoha2.txt.gz")
os.system("mv semeval2020_ulscd_eng/corpus2/token/ccoha2.txt.gz Data/ccoha2_token.txt.gz")
os.system("mv semeval2020_ulscd_eng/corpus1/lemma/ccoha1.txt.gz Data/ccoha1.txt.gz")
os.system("mv semeval2020_ulscd_eng/corpus1/token/ccoha1.txt.gz Data/ccoha1_token.txt.gz")

