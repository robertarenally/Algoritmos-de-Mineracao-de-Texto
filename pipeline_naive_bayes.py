#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Nesse código durante a execução é necessário passar o diretorio do dataset por linha de comando


# In[ ]:


#importanto bibliotecas
import sys #responsável por pegar o diretório por linha de comando
import nltk
import re    
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files 
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV 
from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


#definindo as condições dos casos de testes
steps = [
        ('tfidf', TfidfVectorizer(min_df=2)),
        ('mnb', MultinomialNB())
]
pipe = Pipeline(steps=steps)
params = {
         'mnb__alpha':[1.0000000000000001e-05,9.9999999999999995e-07,0.001,0.01,0.1,1.0]
}
scoring = ['accuracy','f1_micro', 'f1_macro']


# In[ ]:


#Carregando a base de dados
argumento = sys.argv[1:]
dir = 'r'+str(argumento)
dataset = load_files(dir)
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
df.to_csv('resultado_naive_bayes_dataset_classic3.csv')


# In[ ]:


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


# In[ ]:


clf = GridSearchCV(pipe, params, cv=10, n_jobs=-1, verbose=True,scoring=scoring,refit='accuracy',return_train_score=True)
clf.fit(documents,y)
df = pd.DataFrame(clf.cv_results_)
nome_arquivo = 'resultado_naive_bayes_'+str(argumento)+'.csv'
df.to_csv(nome_arquivo)

