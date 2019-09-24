#loading data
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
pos_train = os.listdir("train/pos/")
neg_train = os.listdir("train/neg/")



def create_list(dir,type):
    return_list=[]
    for i in tqdm(range(len(dir))):
        file1 = open("train/"+type+"/" + dir[i])
        try:
            text = file1.read()
            return_list.append(text)
        except UnicodeDecodeError:
            print("error", " ", i)
        file1.close()

    return return_list

pos_train_list= create_list(pos_train,"pos")
neg_train_list= create_list(neg_train,"neg")


import pandas as pd

df=pd.DataFrame(pos_train_list)
target1=[1]*len(pos_train_list)
df["target"]=target1
df=df.rename(columns={0: "text"})

df1=pd.DataFrame(neg_train_list)
target2=[0]*len(neg_train_list)
df1["target"]=target2
df1=df1.rename(columns={0: "text"})

data=pd.concat([df,df1])
x=list(data["text"])


import re
def remove_tags(text):
    s = re.sub(r'<[^>]+>', '', text)
    return s

from nltk.tokenize import word_tokenize

def remove_punctuations(text):
    s = re.sub(r'[^\w\s]', '', text)
    return s


def clean(text):
    vocab=[]
    temp=remove_tags(text)
    text=remove_punctuations(temp)
    for j in word_tokenize(text):
        if (j != ''):
            if not j.islower() and not j.isupper():
                j = j.lower()
            vocab.append(j)
    return vocab


for i in range(0,len(x)):
    x[i]=clean(x[i])


vocab=x[0]
for i in range(1,len(x)):
    vocab.extend(x[i])

vocab=set(vocab)

