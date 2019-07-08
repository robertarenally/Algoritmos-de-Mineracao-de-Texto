#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1 - Importanto as bibliotecas
import sys #responsável por pegar o diretório por linha de comando
import re  
import nltk  
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files 
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


steps = [
        ('tfidf', TfidfVectorizer(min_df=2)),
        ('svc', SVC(degree=2, gamma='auto'))
]
pipe = Pipeline(steps=steps)
params = {
        'svc__C':[100,101,102,103,104,105], 
        'svc__kernel':['linear','poly','rbf']
}
scoring = ['accuracy','f1_micro', 'f1_macro']


# In[ ]:


#Carregando a base de dados
argumento = sys.argv[1:]
dir = 'r'+str(argumento)
dataset = load_files(dir,encoding="ISO-8859-1")
x, y = dataset.data, dataset.target  

#Pré-processamento do texto
documents = []
stemmer = WordNetLemmatizer()

for sen in range(0, len(x)):  
    # Removendo todos os caracteres especiais
    document = re.sub(r'\W', ' ', str(x[sen]))

    # removendo todos os caracteres isolados
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Removendo caracter isolado do ínicio
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

    # Substituindo multiplos espaços por um único espaço
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Convertendo todas as palavras do documento para lower case
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

clf = GridSearchCV(pipe, params, cv=10, n_jobs=-1, verbose=True,scoring=scoring,refit='accuracy',return_train_score=True)
clf.fit(documents,y)
df = pd.DataFrame(clf.cv_results_)
nome_arquivo = 'resultado_svm_'+str(argumento)+'.csv'
df.to_csv(nome_arquivo)

