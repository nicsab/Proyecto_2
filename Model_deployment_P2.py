#!/usr/bin/python
#%cd"C:\Users\nicol\Downloads\Spyder"
            
import pandas as pd
from sklearn.externals import joblib
import numpy as np
import pickle
import sys
import os
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, normalize

import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

Year = 1995
title= 'Major Paynee'
plot = "major benson winifred payne is being discharged from the marines .  payne is a killin '  machine ,  but the wars of the world are no longer fought on the battlefield .  a career marine ,  he has no idea what to do as a civilian ,  so his commander finds him a job  -  commanding officer of a local school ' s jrotc program ,  a bunch or ragtag losers with no hope .  using such teaching tools as live grenades and real bullets ,  payne starts to instill the corp with some hope .  but when payne is recalled to fight in bosnia ,  will he leave the corp that has just started to believe in him ,  or will he find out that killin '  ain ' t much of a livin '  ?"

#Funciones
def remove_punctuation(text):
  import string
  # replacing the punctuations with no space, which in effect deletes the punctuation marks 
  translator = str.maketrans('', '', string.punctuation)
  # return the text stripped of punctuation marks
  return text.translate(translator)

import nltk
nltk.download('stopwords')
#nltk library
sw = stopwords.words('english')  
np.array(sw)
def stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    return " ".join(text)
    
stemmer = SnowballStemmer("english")
def stemming(text):    
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text) 


def predict_proba(Year,title,plot):

    #boostreg = joblib.load(os.path.dirname("C:\Users\nicol\Downloads\Spyder") + '/Bostreg.pkl') 
    #boostreg0 = joblib.load(os.path.dirname(__file__) + '/Bostreg0.pkl')
    #boostreg1 = joblib.load(os.path.dirname(__file__) + '/Bostreg1.pkl')
    #boostreg2 = joblib.load(os.path.dirname(__file__) + '/Bostreg2.pkl')
    
    infile = open('rf_bagging.pkl','rb')
    rf_bagging = pickle.load(infile)
    infile.close()
    
    infile = open('vect_plot.pkl','rb')
    vect_plot = pickle.load(infile)
    infile.close()
    
    infile = open('vect_title.pkl','rb')
    vect_title = pickle.load(infile)
    infile.close()
    
    d = {'year': [Year], 'title': [title], 'plot':[plot]}
    df2 = pd.DataFrame.from_dict(d)
    
    #-- Transformación de variablesbbdd

    df2['time'] = 2019-df2['year'] #Transformación de year
    #df2['titlewords'] = df2['title'].str.split().str.len() #Conteo palabras titulo 'titlewords'
    df2['totalwords'] = df2['plot'].str.split().str.len() #Creación variable totalwords
    df2['Puntuacion']= df2['plot'].str.count('!')+df2['plot'].str.count(',')+df2['plot'].str.count('\.')+df2['plot'].str.count('\?') #Conteo palabras plot 'Puntuacion'
    #Creación variables puntuación
    
    
    #- Remover puntuación
    df2['title'] = df2['title'].apply(remove_punctuation) #Titulo
    df2['plot'] = df2['plot'].apply(remove_punctuation) #Plot 
    
    #- Remover stopwords
    df2['title'] = df2['title'].apply(stopwords) #Titulo
    df2['plot'] = df2['plot'].apply(stopwords) #Plot
    
    #- Stemming
    df2['title'] = df2['title'].apply(stemming) #Titulo
    df2['plot'] = df2['plot'].apply(stemming) #Plot

    #Vectorizar   
    X_dtm = pd.DataFrame(vect_plot.transform(df2['plot']).toarray(), index=df2.index,columns=vect_plot.get_feature_names())
    X_dtm0 = pd.DataFrame(vect_title.transform(df2['title']).toarray(), index=df2.index,columns=vect_title.get_feature_names())

    Time = pd.DataFrame(df2.time, columns=['time'],index=X_dtm.index)
    time2 = Time^2
    Pun =df2[['Puntuacion']]
    w = pd.DataFrame(normalize(df2[['totalwords']],norm='max',axis=0), columns=['Words'],index=X_dtm.index)
    
    Frames = [X_dtm,X_dtm0,Time,time2,Pun,w]
    X_tot = pd.concat(Frames, axis=1)
    
    # Make prediction
    pred = rf_bagging.predict(X_tot)
    cols = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
        'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
        'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
    pred = pd.DataFrame(pred, columns = cols, index = ['probabilidad'])
    pred = pred.transpose()
    
    pred = pred[pred['probabilidad']>=0.5].index.values
        
    return print(pred)


predict_proba(1195, "Major Paynee", "major benson winifred payne is being discharged from the marines .  payne is a killin '  machine ,  but the wars of the world are no longer fought on the battlefield .  a career marine ,  he has no idea what to do as a civilian ,  so his commander finds him a job  -  commanding officer of a local school ' s jrotc program ,  a bunch or ragtag losers with no hope .  using such teaching tools as live grenades and real bullets ,  payne starts to instill the corp with some hope .  but when payne is recalled to fight in bosnia ,  will he leave the corp that has just started to believe in him ,  or will he find out that killin '  ain ' t much of a livin '  ?")


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Por favor ingrese los datos correctamente')
        
    else:

        Year = sys.argv[1]
        title = sys.argv[2]
        plot = sys.argv[3]
        
        pred = predict_proba(Year,title,plot)
        
        print(Year, title, plot)
        print('Gpenero de la película: ', pred)
