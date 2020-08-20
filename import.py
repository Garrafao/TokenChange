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



