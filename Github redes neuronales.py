# -*- coding: utf-8 -*-

"""
Created on Wed Apr 22 20:54:01 2020

@author: Ignacio de Lorenzo
"""


import pandas as pd
import pyodbc 
import os
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from keras import preprocessing
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate, Flatten
from keras.layers import Average
from keras.models import Model
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer


mypath='G:\\Mi unidad\\Investigación\\2021\\ML\\Nueva versión\\'
muestra12=r'G:/Mi unidad/Investigación/2019/ML/Articulo cuadernos/200610 Rexamen muestra.xlsx'
conn = pyodbc.connect(r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=G:\Mi unidad\Investigación\Base Central de Periodística\bcp.accdb;')
os.chdir(mypath)


algo=[]
acctrain=[]
acctest=[]
realtest=[]
sensitest=[]
specifitest=[]


def stemizador (texto):
    stem_words=[]
    texto=str(texto)
    for w in texto.split(" "):
        x = stemmer.stem(w)
        stem_words.append(x)
    return " ".join(stem_words)


def metricas(nombre, y_test, y_pred):
    
    sensitivity=0
    specificity=0
    ypd=0
    accuracy=0
    ypd=pd.DataFrame(zip(y_test,y_pred),columns=["real","prediccion"])
    ypd["eval"]=""
    
    for i in range(0,len(ypd)):
        if ypd.iloc[i]["real"]==1:
            if ypd.iloc[i]["prediccion"]>=0.5:
                ypd.loc[i,"eval"]="tp"
            elif ypd.iloc[i]["prediccion"]<0.5:
               ypd.loc[i,"eval"]="fn"
        elif ypd.iloc[i]["real"]==0:
            if ypd.iloc[i]["prediccion"]>=0.5:
                ypd.loc[i,"eval"]="fp"
            elif ypd.iloc[i]["prediccion"]<0.5:
                ypd.loc[i,"eval"]="tn"
    
    sensitivity=len(ypd[ypd["eval"]=="tp"])/(len(ypd[ypd["eval"]=="tp"])+len(ypd[ypd["eval"]=="fn"]))
    specificity=len(ypd[ypd["eval"]=="tn"])/(len(ypd[ypd["eval"]=="tn"])+len(ypd[ypd["eval"]=="fp"]))
    accuracy=(len(ypd[ypd["eval"]=="tp"])+len(ypd[ypd["eval"]=="tn"]))/(len(ypd[ypd["eval"]=="tp"])+len(ypd[ypd["eval"]=="tn"])+len(ypd[ypd["eval"]=="fp"])+len(ypd[ypd["eval"]=="fn"]))
    
    algo.append(nombre)
    acctest.append(accuracy)
    sensitest.append(sensitivity)
    specifitest.append(specificity)

"""

1-. CARGA DE DATOS


Carga de datos de la base de datos, donde
*** Tabla de libros: todos los datos bibliográficos disponibles.
*** Uninvestigación: filtros de revistas para distintos proyectos.
*** Unificategorias: etiquetas que permiten filtrar los artículos.


"""


print("Cargo todos los datos de la base de datos y arreglo variables")

art=pd.io.sql.read_sql("select * from [Tabla de libros]", 
                            conn)
art.columns= art.columns.str.lower()

art=art.rename(columns={"id":"id_articulo"})

inves=pd.io.sql.read_sql("select * from [Uninvestigacion]", 
                            conn)

inves.columns= inves.columns.str.lower()

inves=inves.rename(columns={"id":"id_invest"})

et=pd.io.sql.read_sql("select * from [unificategoria]", 
                            conn)
et.columns= et.columns.str.lower()

et=et.rename(columns={"id":"id_etiquetas",
                      "idlibro":"idlibro_etiquetas"
                      })


#Para el entrenamiento, se emplearán las revistas originales de De Lorenzo, 2019

print("Incluyo todos los artículos de las revistas originales")

df=pd.merge(art,
            inves[inves["investigación"]==1],
            left_on="revista",
            right_on="revista")

dfn=df

etart=et[et["categoría"]==2]

dfnart=pd.merge(dfn,
                etart,
                left_on="id_articulo",
                right_on="idlibro_etiquetas")

"""

2-. FILTRADO DE LOS ARTÍCULOS

"""

#Primero, artículos académicos.

outlist=[159,27,61,150,81,85,127,136,178,144,12,174,180,179]

print("Quito y etiquetas sobrantes")

eoutlist=et[et.categoría.isin(outlist)]["idlibro_etiquetas"]

dfnartlimp=dfnart.copy()[~dfnart.id_articulo.isin(eoutlist)]

#Artículos anteriores a 2012

df=dfnartlimp[dfnartlimp["fecha de edición"]<=2012]


#Se extraen las clases "Periodística y No Periodística"

etper=et[(et["categoría"]==15)|(et["categoría"]==139)]

dfeti=pd.merge(df,
               etper[["categoría","idlibro_etiquetas"]],
               left_on="id_articulo",
               right_on="idlibro_etiquetas",
               how="left")
dfeti["class_periodistica"]=""

dfeti.loc[dfeti.categoría_y == 15, 'class_periodistica'] = 1
dfeti.loc[dfeti.categoría_y == 139, 'class_periodistica'] = 0
dfeti=dfeti.dropna(subset=["anotaciones"])

"""

PROCESAMIENTO DE TEXTOS

"""


#Eliminación de las stopwords


mistops = stopwords.words('english')+stopwords.words('spanish')+[" u ", "r", "c", "nueva", "parte", "ser", "this", "that", "for", "nuevas", "the", "and", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce", "trece", "catorce", "quince", "hacia", "española", "febrero", "textos", "país", "josé", "páginas", "españolas", "with", "which", "0.12", "from", "vez", "lugar", "estudios", "new", "are", "forma", "españoles", "febrero", "temas", "josé", "clave", "referencias", "día", "días", "central", "clave", "conclusiones", "nueva", "país", "una", "caso", "nuevos", "uso", "with", "cuaderno", "así", "xix", "0.0", "0.04", "sección", "marzo", "meses", "asuntos", "papel", "estudio", "artículo", "trabajo", "este", "juan", "gallega", "españoles", "andalucía", "siglo", "diversidad", "españa", "europa", "nueva", "parte", "ser", "this", "that", "for", "nuevas", "the", "and", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce", "trece", "catorce", "quince", "0.0", "resumen", "communication", "martínez", "evolución", "américa", "resultados", "diferentes",
                                                                 "ahora", "dentro", "hace", "eta", "0.12", "0.08", "0.04", "0.00", "san", "análisis", "español", "poder", "tiempo", "presente", "usos", "ámbito", "use", "históricas", "historia", "reflexión", "acción", "ver", "global", "transformación", "distintos", "reflexión", "concentración", "experiencia", "perspectiva", "futuro", "concepto", "usuarios", "permite", "vida", "grupos", "nuevo", "procesos", "relación", "perspectivas", "necesidad", "modelos", "autor", "partir", "grandes", "recursos", "actual", "visión", "mediante", "problemas", "acceso", "nivel", "diversos", "situación", "conceptos", "importancia", "actual", "verdad", "correo", "publicados", "madrid", "ámbitos", "europea", "sector", "modelo", "entorno", "cambios", "años", "proceso", "its", "0.08", "países", "0.04", "0.00", "sumario", "luis", "constituye", "enero", "introducción", "puede", "analiza", "presenta", "article", "formas", "vez", "través", "are", "forma", "sistema", " u ", "nueva", "parte", "ser", "this", "that", "for", "nuevas", "the", "and", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce", "trece", "catorce", "quince"]

stemmer = SnowballStemmer("spanish")


pat = r'\b(?:{})\b'.format('|'.join(mistops))
dfeti["anotaciones"] = dfeti["anotaciones"].str.lower()
dfeti["anotaciones"] = dfeti["anotaciones"].str.replace('[^\w\s]', '',regex=True)
dfeti["anotaciones"] = dfeti["anotaciones"].str.replace(pat, '',regex=True)
dfeti["anotaciones"] = dfeti["anotaciones"].str.replace(" +", ' ',regex=True)
dfeti["anotaciones"] = dfeti["anotaciones"] .str.replace('\d+', '',regex=True)
dfeti['anotaciones'] = dfeti['anotaciones'].apply(stemizador)


dfeti["título"] = dfeti["título"].str.lower()
dfeti["título"] = dfeti["título"].str.replace('[^\w\s]', '',regex=True)
dfeti["título"] = dfeti["título"].str.replace(pat, '',regex=True)
dfeti["título"] = dfeti["título"].str.replace(" +", ' ',regex=True)
dfeti["título"] = dfeti["título"].str.replace('\d+', '',regex=True)
dfeti['título'] = dfeti['título'].apply(stemizador)

##############################################################################

""" 

ENTRENAMIENTO DEL ALGORITMO

"""

# División en test y training


train, test = train_test_split(dfeti, test_size=0.2)

nwordsa=1000
nwordst=1000
maxlena = 1000
maxlent = 1000

sentencestraina = train["anotaciones"].values
sentencestesta = test["anotaciones"].values

sentencestraint = train["título"].values
sentencestestt = test["título"].values


#Tokenización

tokenizera = Tokenizer(num_words=nwordsa)
tokenizera.fit_on_texts(sentencestraina)

x_trainatok = tokenizera.texts_to_sequences(sentencestraina)
x_testatok = tokenizera.texts_to_sequences(sentencestesta)


tokenizert = Tokenizer(num_words=nwordst)
tokenizert.fit_on_texts(sentencestraint)

x_trainttok = tokenizert.texts_to_sequences(sentencestraint)
x_testttok = tokenizert.texts_to_sequences(sentencestestt)

y = train["class_periodistica"].astype(np.int64).values
yt = test["class_periodistica"].astype(np.int64).values

x_traina = preprocessing.sequence.pad_sequences(x_trainatok, maxlen=maxlena)
x_testa = preprocessing.sequence.pad_sequences(x_testatok, maxlen=maxlena)

x_traint = preprocessing.sequence.pad_sequences(x_trainttok, maxlen=maxlent)
x_testt = preprocessing.sequence.pad_sequences(x_testttok, maxlen=maxlent)


# Construcción de la red neuronal en sí, de títulos (modelt) y abstracts (modela)

input_dima = x_traina.shape[1] 
modela_in = Input(shape=(input_dima))
modela_out=Embedding(1000, 50)(modela_in)
modela_out=Flatten()(modela_out)
modela_out=Dense(1, activation='sigmoid')(modela_out)

modela = Model(modela_in, modela_out)

input_dimt = x_traint.shape[1] 
modelt_in = Input(shape=(input_dimt))
modelt_out=Embedding(1000, 50)(modelt_in)
modelt_out=Flatten()(modelt_out)
modelt_out=Dense(1, activation='sigmoid')(modelt_out)

modelt = Model(modelt_in, modelt_out)


#Combinación de ambas redes neuronales

merged=Average()([modela_out,modelt_out])

mergedd=Model([modela_in,modelt_in],merged)

mergedd.compile(optimizer='adam',
              loss='binary_crossentropy', 
              metrics=['accuracy'])


# Entrenamiento de la red neuronal


mergedd.fit([x_traina, x_traint],
                 y=y,
                 batch_size=100, 
                 epochs=20,
                 validation_data=([x_testa, x_testt], yt),
             verbose=True)


#Prueba sobre el training y test previo a 2012

print("-------------------Predicción sobre el Test<2012")


loss, accuracy = mergedd.evaluate([x_traina, x_traint], y, batch_size=100, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = mergedd.evaluate([x_testa, x_testt], yt, batch_size=50, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


###########Se realiza predicción sobre todo anterior a 2012


sentencestesta = dfeti["anotaciones"].values
sentencestestt = dfeti["título"].values

x_testatok = tokenizera.texts_to_sequences(sentencestesta)
x_testttok = tokenizert.texts_to_sequences(sentencestestt)
x_testa = preprocessing.sequence.pad_sequences(x_testatok, maxlen=maxlena)
x_testt = preprocessing.sequence.pad_sequences(x_testttok, maxlen=maxlent)


dfeti["prediccion"]=mergedd.predict([x_testa, x_testt])


##############################################################################
#Ahora prueba real, de artículos de 2012 a 2020
##############################################################################

#Busca la categoría del test

etest12=et[et["categoría"]==181]

#Ahora proceso los textos como en el traininig

etest12id=list(etest12["idlibro_etiquetas"].drop_duplicates())

m12=dfnartlimp[dfnartlimp["id_articulo"].isin(etest12id)]

m12.columns= m12.columns.str.lower()
m12=m12.copy()
m12=m12.drop_duplicates()
m12["anotacionesres"]=m12["anotaciones"]

pat = r'\b(?:{})\b'.format('|'.join(mistops))
m12["anotaciones"] = m12["anotaciones"].str.lower()
m12["anotaciones"] = m12["anotaciones"].str.replace('[^\w\s]', '',regex=True)
m12["anotaciones"] = m12["anotaciones"].str.replace(pat, '',regex=True)
m12["anotaciones"] = m12["anotaciones"].str.replace(" +", ' ',regex=True)
m12["anotaciones"] = m12["anotaciones"] .str.replace('\d+', '',regex=True)
m12['anotaciones'] = m12['anotaciones'].apply(stemizador)

m12["títulores"]=m12["título"]  

pat = r'\b(?:{})\b'.format('|'.join(mistops))
m12["título"] = m12["título"].str.lower()
m12["título"] = m12["título"].str.replace('[^\w\s]', '',regex=True)
m12["título"] = m12["título"].str.replace(pat, '',regex=True)
m12["título"] = m12["título"].str.replace(" +", ' ',regex=True)
m12["título"] = m12["título"] .str.replace('\d+', '',regex=True)
m12['título'] = m12['título'].apply(stemizador)

m12=pd.merge(m12,
               etper[["categoría","idlibro_etiquetas"]],
               left_on="id_articulo",
               right_on="idlibro_etiquetas",
               how="left")
m12["class_periodistica"]=""

m12.loc[m12.categoría_y == 15, 'class_periodistica'] = 1
m12.loc[m12.categoría_y == 139, 'class_periodistica'] = 0
m12=m12.dropna(subset=["anotaciones"])
m12=m12.dropna(subset=["class_periodistica"])
m12=m12[m12["class_periodistica"]!=""]


sentencestesta = m12["anotaciones"].values
sentencestestt = m12["título"].values

x_testatok = tokenizera.texts_to_sequences(sentencestesta)
x_testttok = tokenizert.texts_to_sequences(sentencestestt)
x_testa = preprocessing.sequence.pad_sequences(x_testatok, maxlen=maxlena)
x_testt = preprocessing.sequence.pad_sequences(x_testttok, maxlen=maxlent)


y = m12["class_periodistica"].astype(np.int64).values

print("-------------------Predicción real test 2012 \n\n")

m12["prediccion"]=mergedd.predict([x_testa, x_testt])

y_pred=m12.copy()[["prediccion"]]
y_pred.loc[y_pred.prediccion > 0.5,"prediccion"] = 1
y_pred.loc[y_pred.prediccion < 0.5,"prediccion"] = 0 


loss, accuracy = mergedd.evaluate([x_testa, x_testt], m12[["class_periodistica"]].astype(np.int64).values, batch_size=50, verbose=False)
print("Test sobre Test>2012:  {:.4f}".format(accuracy))



##############################################################################
##############################################################################

#Predicción sobre 2012 en adelante, no sólo el test

df12=dfnartlimp[dfnartlimp["fecha de edición"]>2012]

dfeti12=df12.drop_duplicates(subset="id_articulo")

dfeti12=dfeti12.dropna(subset=["anotaciones"])

#Vamos a quitarles las stopwords

dfeti12["anotacionesres"]=dfeti12["anotaciones"]

pat = r'\b(?:{})\b'.format('|'.join(mistops))
dfeti12["anotaciones"] = dfeti12["anotaciones"].str.lower()
dfeti12["anotaciones"] = dfeti12["anotaciones"].str.replace('[^\w\s]', '',regex=True)
dfeti12["anotaciones"] = dfeti12["anotaciones"].str.replace(pat, '',regex=True)
dfeti12["anotaciones"] = dfeti12["anotaciones"].str.replace(" +", ' ',regex=True)
dfeti12["anotaciones"] = dfeti12["anotaciones"] .str.replace('\d+', '',regex=True)
dfeti12['anotaciones'] = dfeti12['anotaciones'].apply(stemizador)

dfeti12["títulores"]=dfeti12["título"]  

dfeti12["título"] = dfeti12["título"].str.lower()
dfeti12["título"] = dfeti12["título"].str.replace('[^\w\s]', '',regex=True)
dfeti12["título"] = dfeti12["título"].str.replace(pat, '',regex=True)
dfeti12["título"] = dfeti12["título"].str.replace(" +", ' ',regex=True)
dfeti12["título"] = dfeti12["título"].str.replace('\d+', '',regex=True)
dfeti12['título'] = dfeti12['título'].apply(stemizador)


sentencestesta = dfeti12["anotaciones"].values
sentencestestt = dfeti12["título"].values

x_testatok = tokenizera.texts_to_sequences(sentencestesta)
x_testttok = tokenizert.texts_to_sequences(sentencestestt)
x_testa = preprocessing.sequence.pad_sequences(x_testatok, maxlen=maxlena)
x_testt = preprocessing.sequence.pad_sequences(x_testttok, maxlen=maxlent)


dfeti12["prediccion"]=mergedd.predict([x_testa, x_testt])


dfeti = dfeti.applymap(lambda x: x.encode('unicode_escape').
                 decode('utf-8') if isinstance(x, str) else x)

with pd.ExcelWriter("Predicción de Periodística.xlsx") as writer: 
    dfeti.to_excel(writer, 
                    sheet_name='predoce',
                    index=False)
    dfeti12.to_excel(writer, 
                    sheet_name='posdoce', 
                    index=False)
    m12.to_excel(writer, 
                    sheet_name='Test12', 
                    index=False)

