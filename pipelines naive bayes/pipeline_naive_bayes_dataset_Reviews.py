#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#1 - Importanto as bibliotecas
import re  
import nltk  
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_files 
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV 
from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


#2 - Carregando a base de dados
movie_data = load_files(r"datasets/Reviews",encoding="ISO-8859-1")
x, y = movie_data.data, movie_data.target  


# In[ ]:


#3 - Pré-processamento do texto
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


clf = GridSearchCV(pipe, params, cv=10, n_jobs=-1, verbose=True,scoring=scoring,refit='accuracy',return_train_score=True)
clf.fit(documents,y)


# In[ ]:


df = pd.DataFrame(clf.cv_results_)
df.to_csv('resultado_naive_bayes_dataset_Reviews.csv')

