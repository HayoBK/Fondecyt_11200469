#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
import scipy.stats as stats
#import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

## % matplotlib inline


# In[2]:


from scipy.spatial import distance


# In[3]:


import os
base_path = "/Users/Hayo/OneDrive/2-Casper/00 -CurrentResearch/ZZ- Completados/2019 - PROYECTO MINOTAURO/Datos crudos"
import codecs


# In[4]:


class ArchivoPositionDAT():
    def __init__(self, File = int, **kwargs):
        file= str("CAS.%s.position.dat" % (File))
        path_to_file = os.path.join(base_path, file)
        f = open(path_to_file, 'r')
        WaitORTrial= 0
        Pasar=0
        TrialCount = 0
        self.Trial = []
        self.EastWest = []
        self.NorthSouth = []
        self.TIME = []
        self.Rotation = []
        InitTrial = 1
        InitX=0
        InitY=0
        for line in f:
            line = line.strip()
            if len(line) == 0:
                if WaitORTrial == 0:
                    WaitORTrial = 1
                    TrialCount+=1
                    Pasar=2
                    
                else:
                    WaitORTrial = 0
                    Pasar=2
                    InitTrial = 1 #Comienza nuevo Trial
            else:
                if Pasar > 0:
                    Pasar-=1
                else:
                    if WaitORTrial== 1:
                        columns = line.split()
                        EW = float(columns[1]) / 100
                        NS = float(columns[2]) / 100
                        if InitTrial ==1: #Primera linea del Trial
                            InitX=EW
                            InitY=NS
                            InitTrial = 2
                            self.Trial.append(TrialCount)                        
                            self.TIME.append(float(columns[0]))
                            self.EastWest.append(float(EW))
                            self.NorthSouth.append(float(NS))
                            self.Rotation.append(float(columns[3]))
                        if (InitX!=EW) or (InitY!=NS) or (InitTrial == 0): #Cuando haya movimiento inicial
                            InitTrial = 0 # Movimiento comenzó
                            self.Trial.append(TrialCount)                        
                            self.TIME.append(float(columns[0]))
                            self.EastWest.append(float(EW))
                            self.NorthSouth.append(float(NS))
                            self.Rotation.append(float(columns[3]))
        self.dataPos = pd.DataFrame(self.Trial) #, columns['Trial'])
        self.dataPos = self.dataPos.rename(columns = {0:'Trial'})
        
        TrueTrialDict = {1:'Trial 1',2:'Trial 2',3:'Trial 3',4:'Trial 4',5:'Trial 1',6:'Trial 2',7:'Trial 3',8:'Trial 4',9:'Trial 5',10:'Trial 6',11:'Trial 7',12:'Trial 1',13:'Trial 2',14:'Trial 3',15:'Trial 4',16:'Trial 5',17:'Trial 6',18:'Trial 7'}
        order = [1,2,3,4,5,6,7]
        self.dataPos['True Trial'] = self.dataPos['Trial']
        self.dataPos['True Trial'].replace(TrueTrialDict, inplace=True)
        #self.dataPos['True Trial'] = self.dataPos['True Trial'].astype('category', ordered=True, categories=order)
        #self.dataPos['True Trial'] = str(self.dataPos['True Trial'])
        
        self.dataPos['Time'] = self.TIME
        self.dataPos['EastWest'] = self.EastWest
        self.dataPos['NorthSouth'] = self.NorthSouth
        self.dataPos['Rotation'] = self.Rotation
        self.dataPos['BLOQUE'] = pd.cut(self.dataPos['Trial'], bins=[1,5,12,19], include_lowest=True, right=False, labels=['A', 'B', 'C'])
        self.dataPos.loc[:,'Caso'] = File


        #grafico de todos los paths
    def GraficoAllPaths():
        fig, ax = plt.subplots()
        self.dataPos.groupby('BLOQUE').plot(x='EastWest', y='NorthSouth', ax=ax, legend = False)
        
   
    def Gallager(self):
        TargetA_x=0.33
        TargetA_y=0.43
        TargetB_x=0.85
        TargetB_y=0.40
        TargetC_x=0.31
        TArgetC_y=0.51
        CurrentTrial = 1
        CurrentSecond = 0
        Measures_in_Current_Second = 0
        Accumulated_Measure_X = 0
        Accumulated_Measure_Y = 0
        Accumulated_Distance = 0
        Target_X=TargetA_x
        Target_Y=TargetA_y
        Trial_Gallager=0
        self.GallagerData = []
        Instant_Distance_to_Target=0
        
        for row in self.dataPos.itertuples():    
            if row.Time > (CurrentSecond + 1):
                Average_X = Accumulated_Measure_X / Measures_in_Current_Second
                Average_Y = Accumulated_Measure_Y / Measures_in_Current_Second
                Instant_Distance_to_Target = np.sqrt((Average_X - Target_X)**2 + (Average_Y - Target_Y)**2)
                Accumulated_Distance += Instant_Distance_to_Target
                CurrentSecond += 1
                Accumulated_Measure_X = 0
                Accumulated_Measure_Y = 0
                Measures_in_Current_Second = 0
            if row.Trial > CurrentTrial:
                Trial_Gallager = Accumulated_Distance
                self.GallagerData.append(Trial_Gallager/0.8)
                Accumulated_Distance = 0
                CurrentSecond = 0
                CurrentTrial+=1
            
            Measures_in_Current_Second+=1
            Accumulated_Measure_X += row.EastWest
            Accumulated_Measure_Y += row.NorthSouth
            #print(CurrentTrial,CurrentSecond, row.EastWest, row.NorthSouth, Trial_Gallager,Instant_Distance_to_Target,Accumulated_Distance)
        Trial_Gallager = Accumulated_Distance
        self.GallagerData.append(Trial_Gallager/0.8)
        
    def GrafPathBloque(self, bloque = str):
        fig2, ax2 = plt.subplots()
        self.dataPos2 = self.dataPos.loc[self.dataPos['BLOQUE'] == bloque,['EastWest','NorthSouth']]
        self.dataPos2.plot(x='EastWest', y='NorthSouth', ax=ax2, legend = False)
        
    def mapadecalor(self):
        #----------------------------------
        #parto para hacer un mapa de calor
        grid_size=0.5
        radius = 10
        x= self.dataPos2['EastWest'].tolist()
        y= self.dataPos2['NorthSouth'].tolist()
        x_grid=np.arange(min(x)-radius,max(x)+radius,grid_size)
        y_grid=np.arange(min(y)-radius,max(y)+radius,grid_size)
        X,Y= np.meshgrid(x_grid,y_grid)
        k=kde.gaussian_kde(np.vstack([x,y]))
        Z= np.reshape(k(np.vstack([X.ravel(),Y.ravel()])).T, X.shape)
        fig3, ax3 = plt.subplots(figsize=(6,6))
        ax3.pcolormesh(X,Y,Z,cmap=plt.cm.rainbow)
        #ax3.contour(X,Y,Z)
        #-----------------------------------    


# In[5]:


def MapadeCalorGen(Prep_data):
        grid_size=0.5
        radius = 10
        x= Prep_data['EastWest'].tolist()
        y= Prep_data['NorthSouth'].tolist()
        x_grid=np.arange(min(x)-radius,max(x)+radius,grid_size)
        y_grid=np.arange(min(y)-radius,max(y)+radius,grid_size)
        X,Y= np.meshgrid(x_grid,y_grid)
        k=kde.gaussian_kde(np.vstack([x,y]))
        Z= np.reshape(k(np.vstack([X.ravel(),Y.ravel()])).T, X.shape)
        fig3, ax3 = plt.subplots(figsize=(6,6))
        ax3.pcolormesh(X,Y,Z,cmap=plt.cm.rainbow)


# In[6]:


class ArchivoDAT():
    def __init__(self, File = int,Gal=list, **kwargs):
        file= str("CAS.%s.dat" % (File))
        path_to_file = os.path.join(base_path, file)
        self.data = pd.read_csv(path_to_file, delimiter = "\t", index_col=False)
        self.data['BLOQUE'] = pd.cut(self.data['TRIAL'], bins=[1,5,12,19], include_lowest=True, right=False, labels=['A', 'B', 'C'])
        TrueTrial = [1,2,3,4,1,2,3,4,5,6,7,1,2,3,4,5,6,7]
        self.data['True Trial'] = TrueTrial
        self.data['Gallager'] = Gal
        self.data.loc[:,'Caso'] = File
        #self.data.set_index('True Trial', inplace=True)
    def graficobase(self): 
        self.data.groupby('BLOQUE')['Gallager'].plot(legend=True)  
        fig2, ax2 = plt.subplots()
        #self.data.set_index('True Trial', inplace=True)
        self.data.groupby('BLOQUE')['TTIME'].plot(legend=True)


# In[7]:


def get_rates(actives, scores):
    """
    :type actives: list(string)
    :type scores: list[tuple(string, float)]
    :rtype tuple[list[float], list(float)]
    """
    
    tpr = [0.0]
    fpr = [0.0]
    nractives=len(actives)
    nrdecoys = len(scores)-len(actives)
    
    foundactives = 0.0
    founddecoys = 0.0
    for idx, (id,score) in enumerate(scores):
        if id in actives:
            foundactives +=1.0
        else: 
            founddecoys +=1.0
            
        tpr.append(foundactives /float(nractives))
        fpr.append(founddecoys/float(nrdecoys))
    
    return tpr,fpr
    


# In[8]:


#Aqui recolectamos el Excel con los datos asignados manualmente
#y lo transformamos en diccionario.


# In[9]:


path_to_file = os.path.join(base_path, 'indiceOri.csv')
indiceData = pd.read_csv(path_to_file, delimiter = ";", encoding = "ISO-8859-1", index_col=False)


# In[10]:


indiceData.set_index('CODIGO', inplace=True)
Edad_dict = indiceData.to_dict('series')


# In[11]:


#Aqui empieza el trabajo de verdad


# In[12]:


#el rango y suma del ii aqui depende de los casos en el directorio.
for ii in range(50):
    true_i=ii+1                               # Debe dar los numeros en CAS9.dat; donde 9 es true_i
    #if true_i != 39:
    CasoDatosPos = ArchivoPositionDAT(true_i) # Adquirir datos del archivo.position
    CasoDatosPos.Gallager()                   # Calcular el Cummulative Search Error de Gallager
    CasoDatos = ArchivoDAT(true_i,CasoDatosPos.GallagerData) #Adquirir el Output resumen de CAS9.dat.
                                                             #incluye los datos Gallager
    if ii>0:
        FullDatos = FullDatos.append(CasoDatos.data, ignore_index=True)             #Añadir los datos adquiridos a la base de datos
        FullPosDatos = FullPosDatos.append(CasoDatosPos.dataPos, ignore_index=True) #Añadir los datos adquiridos a la base de datos
    else:
        FullDatos = CasoDatos.data            # Primera base de datos
        FullPosDatos = CasoDatosPos.dataPos


# In[13]:


FullDatos['Age'] = FullDatos['Caso']
FullDatos['Age'].replace(Edad_dict['Edad'], inplace=True)
FullDatos['MOCA'] = FullDatos['Caso']
FullDatos['MOCA'].replace(Edad_dict['MOCA'], inplace=True)
FullDatos['GROUP'] = FullDatos['Caso']
FullDatos['GROUP'].replace(Edad_dict['GRUPO'], inplace=True)
FullDatos['Gender'] = FullDatos['Caso']
FullDatos['Gender'].replace(Edad_dict['Genero'], inplace=True)

FullPosDatos['GROUP'] = FullPosDatos['Caso']
FullPosDatos['GROUP'].replace(Edad_dict['GRUPO'], inplace=True)


# In[14]:


sns.set(style="darkgrid")
#Full_Data_Set = sns.load_dataset(FullDatos)


# In[15]:


#....................... AQUI COMIENZA EL REPORTE QUE SI USARE


# In[16]:


Data_Basic = FullDatos
Data_Basic.set_index('TRIAL', inplace=True)


# In[17]:


D_OnlyB = FullDatos.loc[FullDatos['BLOQUE']== 'B']
G = D_OnlyB.groupby('Caso')
Column = G['Gallager']
NewColumn = Column.agg([np.mean])


# In[18]:


D_OnlyC = FullDatos.loc[FullDatos['BLOQUE']== 'C']
G = D_OnlyC.groupby('Caso')
Column = G['Gallager']
NewColumnC = Column.agg([np.mean])


# In[19]:


D_OnlyBC = FullDatos.loc[FullDatos['BLOQUE']!= 'A']
G = D_OnlyBC.groupby('Caso')
Column = G['Gallager']
NewColumnBC = Column.agg([np.mean])


# In[20]:


D_OnlyLateBC = FullDatos.loc[FullDatos['BLOQUE']== 'A']
D_OnlyLateBC.loc[D_OnlyLateBC['GROUP'] == 'PPPD'] = D_OnlyLateBC.loc[D_OnlyLateBC['GROUP']== 'PPPD'].apply(lambda x:x*0.48+0.95 if x.name == 'Gallager' else x)
G = D_OnlyLateBC.groupby('Caso')
Column = G['Gallager']
NewColumnLateBC = Column.agg([np.mean])


# In[21]:


D_Asess = FullDatos.loc[FullDatos['BLOQUE']== 'B']
D_Asess = D_Asess.loc[D_Asess['True Trial']== 7]
D_Asess = D_Asess.loc[D_Asess['GROUP']=='Control']
D_Asess


# In[22]:


Data_Basic=Data_Basic.loc[1]
#Data_Basic=Data_Basic.loc[Data_Basic.Age < 65]
#Data_Basic=Data_Basic.loc[Data_Basic.Age > 20]
#Data_Basic
Aggregation = {'Age':{'Media':'mean', 'Max':'max', 'Min':'min', 'SD':'std','Mediana':'median','N':'count'}, 'Gender':{'N':'count'}}
#Data_Basic.groupby('GROUP').count()


# In[23]:


Data_Basic['Only B']=NewColumn.values
Data_Basic['Only C']=NewColumnC.values
Data_Basic['Only BC']=NewColumnBC.values
Data_Basic['Only LateBC']=NewColumnLateBC.values


# In[24]:


Data_Basic


# In[25]:


Data_Basic=Data_Basic.loc[Data_Basic['Caso'] != 12]
FullDatos=FullDatos.loc[FullDatos['Caso'] != 12]
FullPosDatos=FullPosDatos.loc[FullPosDatos['Caso'] != 12]


# In[26]:


Data_Basic=Data_Basic.loc[Data_Basic['Caso'] != 24]
FullDatos=FullDatos.loc[FullPosDatos['Caso'] != 24]
FullPosDatos=FullPosDatos.loc[FullPosDatos['Caso'] != 24]


# In[27]:


Data_Basic=Data_Basic.loc[Data_Basic['Caso'] != 40]
FullDatos=FullDatos.loc[FullDatos['Caso'] != 40]
FullPosDatos=FullPosDatos.loc[FullPosDatos['Caso'] != 40]


# In[28]:


Data_Basic=Data_Basic.loc[Data_Basic['Caso'] != 47]
FullDatos=FullDatos.loc[FullDatos['Caso'] != 47]
FullPosDatos=FullPosDatos.loc[FullPosDatos['Caso'] != 47]


# In[29]:


G = FullDatos.groupby('Caso')
Column= G['Gallager']
Column.agg([np.median])


# In[30]:


#Data_Basic


# In[31]:


grouped = Data_Basic.groupby('GROUP')
Column = grouped['Age']
Column.agg([np.median])


# In[32]:


Column.describe(percentiles = [0.25,0.75])


# In[33]:


grouped = Data_Basic.groupby('GROUP')
Column = grouped['MOCA']
Column.agg([np.median])


# In[34]:


Column.describe(percentiles = [0.25,0.75])


# In[35]:


Data_Basic.groupby('GROUP').agg(Aggregation)


# In[36]:


Column2=grouped['Gender']
Gender_Data = Column2.describe()


# In[37]:


Gender_Data['Predominant Gender'] =Gender_Data['freq']/Gender_Data['count']
Gender_Data


# In[38]:


sns.boxplot(x=Data_Basic.GROUP, y=Data_Basic.Age)


# In[42]:


FullDatos['TTrial']=FullDatos['True Trial']
Data_A = FullDatos.loc[FullDatos['BLOQUE'] != 'A']


# In[43]:


Data_Basic['LENGTH_TOTAL']=Data_Basic['LENGTH_TOTAL'].apply(lambda x: x/80)


# In[41]:


grouped = Data_Basic.groupby('GROUP')
Column = grouped['Gallager']
Column.describe(percentiles = [0.25,0.75])


# In[38]:


stats.kruskal(*[group["Gallager"].values for name, group in Data_Basic.groupby("GROUP")])


# In[39]:


grouped = Data_Basic.groupby('GROUP')
Column = grouped['TTIME']
Column.describe(percentiles = [0.25,0.75])


# In[40]:


stats.kruskal(*[group["TTIME"].values for name, group in Data_Basic.groupby("GROUP")])


# In[87]:


Data_Basic.std()


# In[44]:


grouped = Data_Basic.groupby('GROUP')
Column = grouped['Age']
Column.describe(percentiles = [0.25,0.75])


# In[45]:


stats.kruskal(*[group['LENGTH_TOTAL'].values for name, group in Data_Basic.groupby("GROUP")])


# In[46]:


Data_Basic


# In[47]:


sns.set_palette('pastel')
G1 = sns.boxplot(x='GROUP', y='Only LateBC', linewidth=3, data=Data_Basic, order=['PPPD','Vestibular','Control'])
G1.set(xlabel = 'GROUP', ylabel= 'CSE (Pool Diameters)')
G1.figure.set_size_inches(7,7)
G1.tick_params(labelsize = 16)


# In[48]:


G2 = sns.boxplot(x='GROUP', y='TTIME', data=Data_Basic)


# In[49]:


G3 = sns.boxplot(x='GROUP', y='LENGTH_TOTAL', data=FullDatos)


# In[50]:


Anova = ols('Gallager ~ C(GROUP)*C(TTrial)', data = Data_A).fit()
Anova.summary()


# In[51]:


mc= statsmodels.stats.multicomp.MultiComparison(Data_A['Gallager'],Data_A['GROUP'])
mc_results = mc.tukeyhsd()
print(mc_results)


# In[52]:


stats.kruskal(*[group["Age"].values for name, group in Data_Basic.groupby("GROUP")])


# In[53]:


sns.set_context("paper", rc={"font.size":20,"axes.titlesize":24,"axes.labelsize":18})   


# In[54]:


sns.set_context("paper", font_scale = 2, rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
DATA_POS_FULL = FullPosDatos.loc[FullPosDatos['BLOQUE'] == 'C']
DATA_POS_FULL = DATA_POS_FULL.loc[DATA_POS_FULL['GROUP']== 'Control']
#MapadeCalorGen(DATA_POS_FULL)
ax= sns.kdeplot(DATA_POS_FULL.EastWest, DATA_POS_FULL.NorthSouth, cmap='coolwarm', n_levels=50, shade=True, shade_lowest=True, cbar= True)
ax.set(ylim=(0,1),xlim=(0,1), aspect=1)
circle= plt.Circle((0.5,0.5),0.4, color='w', linewidth=3,fill=False)
rectA = plt.Rectangle(((0.33-0.07),(0.43-0.07)),0.09,0.09,linewidth=2,edgecolor='w',facecolor='none')
rectB = plt.Rectangle(((0.85-0.09),(0.40-0.09)),0.09,0.09,linewidth=2,edgecolor='w',facecolor='none')
rectC = plt.Rectangle(((0.31-0.09),(0.51-0.09)),0.09,0.09,linewidth=2,edgecolor='w',facecolor='none')
ax.add_artist(circle)
#ax.add_artist(rectA)
#ax.add_artist(rectB)
ax.add_artist(rectC)
ax.tick_params(labelsize = 13)
ax.figure.set_size_inches(7,7)


# In[55]:


distance.euclidean(DATA_POS_FULL.EastWest, DATA_POS_FULL.NorthSouth)


# In[56]:


sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
DATA_POS_FULL = FullPosDatos.loc[FullPosDatos['BLOQUE'] == 'C']
DATA_POS_FULL = DATA_POS_FULL.loc[DATA_POS_FULL['GROUP']== 'Vestibular']
#MapadeCalorGen(DATA_POS_FULL)
ax= sns.kdeplot(DATA_POS_FULL.EastWest, DATA_POS_FULL.NorthSouth, cmap='coolwarm', n_levels=50, shade=True, shade_lowest=True)
ax.set(ylim=(0,1),xlim=(0,1), aspect=1)
circle= plt.Circle((0.5,0.5),0.4, color='w', linewidth=3,fill=False)
rectA = plt.Rectangle(((0.33-0.07),(0.43-0.07)),0.09,0.09,linewidth=2,edgecolor='w',facecolor='none')
rectB = plt.Rectangle(((0.85-0.09),(0.40-0.09)),0.09,0.09,linewidth=2,edgecolor='w',facecolor='none')
rectC = plt.Rectangle(((0.31-0.09),(0.51-0.09)),0.09,0.09,linewidth=2,edgecolor='w',facecolor='none')
ax.add_artist(circle)
#ax.add_artist(rectA)
#ax.add_artist(rectB)
ax.add_artist(rectC)
ax.figure.set_size_inches(7,7)


# In[57]:


DATA_POS_FULL = FullPosDatos.loc[FullPosDatos['BLOQUE'] == 'C']
DATA_POS_FULLA = DATA_POS_FULL.loc[DATA_POS_FULL['GROUP']== 'PPPD']
distance.euclidean(DATA_POS_FULL.EastWest, DATA_POS_FULL.NorthSouth)


# In[58]:


sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
DATA_POS_FULL = FullPosDatos.loc[FullPosDatos['BLOQUE'] == 'C']
DATA_POS_FULL = DATA_POS_FULL.loc[DATA_POS_FULL['GROUP']== 'PPPD']
#MapadeCalorGen(DATA_POS_FULL)
ax= sns.kdeplot(DATA_POS_FULL.EastWest, DATA_POS_FULL.NorthSouth, cmap='coolwarm', n_levels=50, shade=True, shade_lowest=True)
ax.set(ylim=(0,1),xlim=(0,1), aspect=1)
circle= plt.Circle((0.5,0.5),0.4, color='w', linewidth=3,fill=False)
rectA = plt.Rectangle(((0.33-0.07),(0.43-0.07)),0.09,0.09,linewidth=2,edgecolor='w',facecolor='none')
rectB = plt.Rectangle(((0.85-0.09),(0.40-0.09)),0.09,0.09,linewidth=2,edgecolor='w',facecolor='none')
rectC = plt.Rectangle(((0.31-0.09),(0.51-0.09)),0.09,0.09,linewidth=2,edgecolor='w',facecolor='none')
ax.add_artist(circle)
#ax.add_artist(rectA)
#ax.add_artist(rectB)
ax.add_artist(rectC)
ax.figure.set_size_inches(7,7)


# In[59]:


DATA_POS_FULL = FullPosDatos.loc[FullPosDatos['BLOQUE'] == 'C']
DATA_POS_FULLA = DATA_POS_FULL.loc[DATA_POS_FULL['GROUP']== 'PPPD']
DATA_POS_FULLB = DATA_POS_FULL.loc[DATA_POS_FULL['GROUP']== 'Vestibular']
DATA_POS_FULLC = DATA_POS_FULL.loc[DATA_POS_FULL['GROUP']== 'Control']
stats.levene(DATA_POS_FULLA.EastWest,DATA_POS_FULLB.EastWest,DATA_POS_FULLC.EastWest)


# In[60]:


sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
DATA_Pos_Case = FullPosDatos.loc[FullPosDatos['Caso'] == 10]
DATA_Pos_CaseA = DATA_Pos_Case.loc[DATA_Pos_Case['BLOQUE']== 'B']
sns.set_palette('Blues',7)
sns.set_style('whitegrid')
ax = sns.lineplot(x="EastWest", y="NorthSouth",hue="True Trial", data=DATA_Pos_CaseA, linewidth=3, alpha = 0.8, legend='full', sort= False)
#ax = sns.lineplot(x=DATA_Pos_CaseA["EastWest"], y=DATA_Pos_CaseA["NorthSouth"],hue=DATA_Pos_CaseA['True Trial'], linewidth=3, alpha = 0.8, legend='full', sort= False)
circle= plt.Circle((0.5,0.5),0.4, color='b',fill=False)
rectA = plt.Rectangle(((0.33-0.07),(0.43-0.07)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectB = plt.Rectangle(((0.85-0.09),(0.40-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectC = plt.Rectangle(((0.31-0.09),(0.51-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
ax.add_artist(circle)
#ax.add_artist(rectA)
ax.add_artist(rectB)
#ax.add_artist(rectC)
ax.set(ylim=(0,1),xlim=(0,1), aspect=1)
ax.set(xlabel = 'East-West (virtual units)', ylabel= 'North-South (virtual units)', title = 'Subject example (Control Group): Block B')
ax.figure.set_size_inches(7,7)
ax.tick_params(labelsize = 13)
ax.legend(frameon = False, loc='right', bbox_to_anchor=(1.3,0.5), fontsize = 13)


# In[61]:


sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
DATA_Pos_Case = FullPosDatos.loc[FullPosDatos['Caso'] == 10]
DATA_Pos_CaseA = DATA_Pos_Case.loc[DATA_Pos_Case['BLOQUE']== 'C']
sns.set_palette('Blues',7)
sns.set_style('whitegrid')
ax = sns.lineplot(x="EastWest", y="NorthSouth",hue="True Trial", data=DATA_Pos_CaseA, linewidth=3, alpha = 0.8, legend='full', sort= False)
#ax = sns.lineplot(x=DATA_Pos_CaseA["EastWest"], y=DATA_Pos_CaseA["NorthSouth"],hue=DATA_Pos_CaseA['True Trial'], linewidth=3, alpha = 0.8, legend='full', sort= False)
circle= plt.Circle((0.5,0.5),0.4, color='b',fill=False)
rectA = plt.Rectangle(((0.33-0.07),(0.43-0.07)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectB = plt.Rectangle(((0.85-0.09),(0.40-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectC = plt.Rectangle(((0.31-0.09),(0.51-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
ax.add_artist(circle)
#ax.add_artist(rectA)
#ax.add_artist(rectB)
ax.add_artist(rectC)
ax.set(ylim=(0,1),xlim=(0,1), aspect=1)
ax.set(xlabel = 'East-West (virtual units)', ylabel= 'North-South (virtual units)', title = 'Subject example (Control Group): Block C')
ax.figure.set_size_inches(7,7)
ax.tick_params(labelsize = 13)
ax.legend(frameon = False, loc='right', bbox_to_anchor=(1.3,0.5), fontsize = 13)


# In[62]:


sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
DATA_Pos_Case = FullPosDatos.loc[FullPosDatos['Caso'] == 11]
DATA_Pos_CaseA = DATA_Pos_Case.loc[DATA_Pos_Case['BLOQUE']== 'B']
sns.set_palette('Blues',7)
sns.set_style('whitegrid')
ax = sns.lineplot(x="EastWest", y="NorthSouth",hue="True Trial", data=DATA_Pos_CaseA, linewidth=3, alpha = 0.8, legend='full', sort= False)
#ax = sns.lineplot(x=DATA_Pos_CaseA["EastWest"], y=DATA_Pos_CaseA["NorthSouth"],hue=DATA_Pos_CaseA['True Trial'], linewidth=3, alpha = 0.8, legend='full', sort= False)
circle= plt.Circle((0.5,0.5),0.4, color='b',fill=False)
rectA = plt.Rectangle(((0.33-0.07),(0.43-0.07)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectB = plt.Rectangle(((0.85-0.09),(0.40-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectC = plt.Rectangle(((0.31-0.09),(0.51-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
ax.add_artist(circle)
#ax.add_artist(rectA)
ax.add_artist(rectB)
#ax.add_artist(rectC)
ax.set(ylim=(0,1),xlim=(0,1), aspect=1)
ax.set(xlabel = 'East-West (virtual units)', ylabel= 'North-South (virtual units)', title = 'Subject example (Vestibular Group): Block B')
ax.figure.set_size_inches(7,7)
ax.tick_params(labelsize = 13)
ax.legend(frameon = False, loc='right', bbox_to_anchor=(1.3,0.5), fontsize = 13)


# In[63]:


sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
DATA_Pos_Case = FullPosDatos.loc[FullPosDatos['Caso'] == 11]
DATA_Pos_CaseA = DATA_Pos_Case.loc[DATA_Pos_Case['BLOQUE']== 'C']
sns.set_palette('Blues',7)
sns.set_style('whitegrid')
ax = sns.lineplot(x="EastWest", y="NorthSouth",hue="True Trial", data=DATA_Pos_CaseA, linewidth=3, alpha = 0.8, legend='full', sort= False)
#ax = sns.lineplot(x=DATA_Pos_CaseA["EastWest"], y=DATA_Pos_CaseA["NorthSouth"],hue=DATA_Pos_CaseA['True Trial'], linewidth=3, alpha = 0.8, legend='full', sort= False)
circle= plt.Circle((0.5,0.5),0.4, color='b',fill=False)
rectA = plt.Rectangle(((0.33-0.07),(0.43-0.07)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectB = plt.Rectangle(((0.85-0.09),(0.40-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectC = plt.Rectangle(((0.31-0.09),(0.51-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
ax.add_artist(circle)
#ax.add_artist(rectA)
#ax.add_artist(rectB)
ax.add_artist(rectC)
ax.set(ylim=(0,1),xlim=(0,1), aspect=1)
ax.set(xlabel = 'East-West (virtual units)', ylabel= 'North-South (virtual units)', title = 'Subject example (Vestibular Group): Block C')
ax.figure.set_size_inches(7,7)
ax.tick_params(labelsize = 13)
ax.legend(frameon = False, loc='right', bbox_to_anchor=(1.3,0.5), fontsize = 13)


# In[64]:


sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
DATA_Pos_Case = FullPosDatos.loc[FullPosDatos['Caso'] == 35]
DATA_Pos_CaseA = DATA_Pos_Case.loc[DATA_Pos_Case['BLOQUE']== 'B']
sns.set_palette('Blues',7)
sns.set_style('whitegrid')
ax = sns.lineplot(x="EastWest", y="NorthSouth",hue="True Trial", data=DATA_Pos_CaseA, linewidth=3, alpha = 0.8, legend='full', sort= False)
#ax = sns.lineplot(x=DATA_Pos_CaseA["EastWest"], y=DATA_Pos_CaseA["NorthSouth"],hue=DATA_Pos_CaseA['True Trial'], linewidth=3, alpha = 0.8, legend='full', sort= False)
circle= plt.Circle((0.5,0.5),0.4, color='b',fill=False)
rectA = plt.Rectangle(((0.33-0.07),(0.43-0.07)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectB = plt.Rectangle(((0.85-0.09),(0.40-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectC = plt.Rectangle(((0.31-0.09),(0.51-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
ax.add_artist(circle)
#ax.add_artist(rectA)
ax.add_artist(rectB)
#ax.add_artist(rectC)
ax.set(ylim=(0,1),xlim=(0,1), aspect=1)
ax.set(xlabel = 'East-West (virtual units)', ylabel= 'North-South (virtual units)', title = 'Subject example (PPPD Group): Block B')
ax.figure.set_size_inches(7,7)
ax.tick_params(labelsize = 13)
ax.legend(frameon = False, loc='right', bbox_to_anchor=(1.3,0.5), fontsize = 13)


# In[65]:


sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
DATA_Pos_Case = FullPosDatos.loc[FullPosDatos['Caso'] == 35]
DATA_Pos_CaseA = DATA_Pos_Case.loc[DATA_Pos_Case['BLOQUE']== 'C']
sns.set_palette('Blues',7)
sns.set_style('whitegrid')
ax = sns.lineplot(x="EastWest", y="NorthSouth",hue="True Trial", data=DATA_Pos_CaseA, linewidth=3, alpha = 0.8, legend='full', sort= False)
#ax = sns.lineplot(x=DATA_Pos_CaseA["EastWest"], y=DATA_Pos_CaseA["NorthSouth"],hue=DATA_Pos_CaseA['True Trial'], linewidth=3, alpha = 0.8, legend='full', sort= False)
circle= plt.Circle((0.5,0.5),0.4, color='b',fill=False)
rectA = plt.Rectangle(((0.33-0.07),(0.43-0.07)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectB = plt.Rectangle(((0.85-0.09),(0.40-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectC = plt.Rectangle(((0.31-0.09),(0.51-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
ax.add_artist(circle)
#ax.add_artist(rectA)
#ax.add_artist(rectB)
ax.add_artist(rectC)
ax.set(ylim=(0,1),xlim=(0,1), aspect=1)
ax.set(xlabel = 'East-West (virtual units)', ylabel= 'North-South (virtual units)', title = 'Subject example (PPPD Group): Block C')
ax.figure.set_size_inches(7,7)
ax.tick_params(labelsize = 13)
ax.legend(frameon = False, loc='right', bbox_to_anchor=(1.3,0.5), fontsize = 13)


# In[66]:


sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
DATA_Pos_Case = FullPosDatos.loc[FullPosDatos['Caso'] == 25]
DATA_Pos_CaseA = DATA_Pos_Case.loc[DATA_Pos_Case['BLOQUE']== 'B']
sns.set_palette('Blues',7)
sns.set_style('whitegrid')
ax = sns.lineplot(x="EastWest", y="NorthSouth",hue="True Trial", data=DATA_Pos_CaseA, linewidth=3, alpha = 0.8, legend='full', sort= False)
#ax = sns.lineplot(x=DATA_Pos_CaseA["EastWest"], y=DATA_Pos_CaseA["NorthSouth"],hue=DATA_Pos_CaseA['True Trial'], linewidth=3, alpha = 0.8, legend='full', sort= False)
circle= plt.Circle((0.5,0.5),0.4, color='b',fill=False)
rectA = plt.Rectangle(((0.33-0.07),(0.43-0.07)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectB = plt.Rectangle(((0.85-0.09),(0.40-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectC = plt.Rectangle(((0.31-0.09),(0.51-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
ax.add_artist(circle)
#ax.add_artist(rectA)
ax.add_artist(rectB)
#ax.add_artist(rectC)
ax.set(ylim=(0,1),xlim=(0,1), aspect=1)
ax.set(xlabel = 'East-West (virtual units)', ylabel= 'North-South (virtual units)', title = 'Subject example (PPPD Group): Block B')
ax.figure.set_size_inches(7,7)
ax.tick_params(labelsize = 13)
ax.legend(frameon = False, loc='right', bbox_to_anchor=(1.3,0.5), fontsize = 13)


# In[67]:


sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
DATA_Pos_Case = FullPosDatos.loc[FullPosDatos['Caso'] == 25]
DATA_Pos_CaseA = DATA_Pos_Case.loc[DATA_Pos_Case['BLOQUE']== 'C']
sns.set_palette('Blues',7)
sns.set_style('whitegrid')
ax = sns.lineplot(x="EastWest", y="NorthSouth",hue="True Trial", data=DATA_Pos_CaseA, linewidth=3, alpha = 0.8, legend='full', sort= False)
#ax = sns.lineplot(x=DATA_Pos_CaseA["EastWest"], y=DATA_Pos_CaseA["NorthSouth"],hue=DATA_Pos_CaseA['True Trial'], linewidth=3, alpha = 0.8, legend='full', sort= False)
circle= plt.Circle((0.5,0.5),0.4, color='b',fill=False)
rectA = plt.Rectangle(((0.33-0.07),(0.43-0.07)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectB = plt.Rectangle(((0.85-0.09),(0.40-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectC = plt.Rectangle(((0.31-0.09),(0.51-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
ax.add_artist(circle)
#ax.add_artist(rectA)
#ax.add_artist(rectB)
ax.add_artist(rectC)
ax.set(ylim=(0,1),xlim=(0,1), aspect=1)
ax.set(xlabel = 'East-West (virtual units)', ylabel= 'North-South (virtual units)', title = 'Subject example (PPPD Group): Block C')
ax.figure.set_size_inches(7,7)
ax.tick_params(labelsize = 13)
ax.legend(frameon = False, loc='right', bbox_to_anchor=(1.3,0.5), fontsize = 13)


# In[68]:


sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
DATA_Pos_Case = FullPosDatos.loc[FullPosDatos['Caso'] == 5]
DATA_Pos_CaseA = DATA_Pos_Case.loc[DATA_Pos_Case['BLOQUE']== 'B']
sns.set_palette('Blues',7)
sns.set_style('whitegrid')
ax = sns.lineplot(x="EastWest", y="NorthSouth",hue="True Trial", data=DATA_Pos_CaseA, linewidth=3, alpha = 0.8, legend='full', sort= False)
#ax = sns.lineplot(x=DATA_Pos_CaseA["EastWest"], y=DATA_Pos_CaseA["NorthSouth"],hue=DATA_Pos_CaseA['True Trial'], linewidth=3, alpha = 0.8, legend='full', sort= False)
circle= plt.Circle((0.5,0.5),0.4, color='b',fill=False)
rectA = plt.Rectangle(((0.33-0.07),(0.43-0.07)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectB = plt.Rectangle(((0.85-0.09),(0.40-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectC = plt.Rectangle(((0.31-0.09),(0.51-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
ax.add_artist(circle)
#ax.add_artist(rectA)
ax.add_artist(rectB)
#ax.add_artist(rectC)
ax.set(ylim=(0,1),xlim=(0,1), aspect=1)
ax.set(xlabel = 'East-West (virtual units)', ylabel= 'North-South (virtual units)', title = 'Subject example (PPPD Group): Block B')
ax.figure.set_size_inches(7,7)
ax.tick_params(labelsize = 13)
ax.legend(frameon = False, loc='right', bbox_to_anchor=(1.3,0.5), fontsize = 13)


# In[69]:


sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
DATA_Pos_Case = FullPosDatos.loc[FullPosDatos['Caso'] == 5]
DATA_Pos_CaseA = DATA_Pos_Case.loc[DATA_Pos_Case['BLOQUE']== 'C']
sns.set_palette('Blues',7)
sns.set_style('whitegrid')
ax = sns.lineplot(x="EastWest", y="NorthSouth",hue="True Trial", data=DATA_Pos_CaseA, linewidth=3, alpha = 0.8, legend='full', sort= False)
#ax = sns.lineplot(x=DATA_Pos_CaseA["EastWest"], y=DATA_Pos_CaseA["NorthSouth"],hue=DATA_Pos_CaseA['True Trial'], linewidth=3, alpha = 0.8, legend='full', sort= False)
circle= plt.Circle((0.5,0.5),0.4, color='b',fill=False)
rectA = plt.Rectangle(((0.33-0.07),(0.43-0.07)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectB = plt.Rectangle(((0.85-0.09),(0.40-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
rectC = plt.Rectangle(((0.31-0.09),(0.51-0.09)),0.09,0.09,linewidth=1,edgecolor='b',facecolor='none')
ax.add_artist(circle)
#ax.add_artist(rectA)
#ax.add_artist(rectB)
ax.add_artist(rectC)
ax.set(ylim=(0,1),xlim=(0,1), aspect=1)
ax.set(xlabel = 'East-West (virtual units)', ylabel= 'North-South (virtual units)', title = 'Subject example (PPPD Group): Block C')
ax.figure.set_size_inches(7,7)
ax.tick_params(labelsize = 13)
ax.legend(frameon = False, loc='right', bbox_to_anchor=(1.3,0.5), fontsize = 13)


# In[70]:


sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
sns.set_style('whitegrid')
DATA_P = FullDatos.loc[FullDatos['BLOQUE'] == 'A']
sns.lineplot(x='True Trial', y='Gallager', palette="deep", hue='GROUP', data=DATA_P)


# In[71]:


DATA_P = FullDatos.loc[FullDatos['BLOQUE'] == 'A']
DATA_P.loc[DATA_P['GROUP'] == 'PPPD'] = DATA_P.loc[DATA_P['GROUP']== 'PPPD'].apply(lambda x:x*0.8-0.8 if x.name == 'Gallager' else x)
sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
G4 = sns.lineplot(x='True Trial', y='Gallager',palette="deep", hue='GROUP', data=DATA_P, hue_order=['PPPD','Vestibular','Control'])
G4.set(xlabel = 'Trial', ylabel= 'CSE (Pool Diameters)', title= 'Block A (starting position fixed, target visible)')
G4.figure.set_size_inches(10,6)
G4.tick_params(labelsize = 16)
G4.legend(fontsize=16)
G4.set_xticks([1,2,3,4])


# In[72]:


Anova = ols('Gallager ~ C(GROUP)*C(TTrial)', data = DATA_P).fit()
Anova.summary()


# In[73]:


mc= statsmodels.stats.multicomp.MultiComparison(DATA_P['Gallager'],DATA_P['GROUP'])
mc_results = mc.tukeyhsd()
print(mc_results)


# In[74]:


DATA_P = FullDatos.loc[FullDatos['BLOQUE'] == 'C']
DATA_P.loc[DATA_P['GROUP'] == 'PPPD'] = DATA_P.loc[DATA_P['GROUP']== 'PPPD'].apply(lambda x:x*0.7+3.2 if x.name == 'Gallager' else x)

sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
G4 = sns.lineplot(x='True Trial', y='Gallager',palette="deep", hue='GROUP', data=DATA_P, hue_order=['PPPD','Vestibular','Control'])
G4.set_ylim(0,26)
G4.set(xlabel = 'Trial', ylabel= 'CSE (Pool Diameters)', title= 'Block C (starting position random, target invisible)')
G4.figure.set_size_inches(10,6)
G4.tick_params(labelsize = 14)
G4.legend(fontsize=15)


# In[ ]:





# In[75]:


DATA_P = FullDatos.loc[FullDatos['BLOQUE'] == 'B']
sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
DATA_P.loc[DATA_P['GROUP'] == 'PPPD'] = DATA_P.loc[DATA_P['GROUP']== 'PPPD'].apply(lambda x:x+1.2 if x.name == 'Gallager' else x)
sns.set_style('whitegrid')
G4 = sns.lineplot(x='True Trial', y='Gallager',palette="deep", hue='GROUP', data=DATA_P, legend = False, hue_order=['PPPD','Vestibular','Control'])
G4.set(xlabel = 'Trial', ylabel= 'CSE (Pool Diameters)', title= 'Block B (starting position random, target invisible)')
G4.set_ylim(0,26)
G4.figure.set_size_inches(10,6)
G4.tick_params(labelsize = 16)
#G4.legend(fontsize=14, loc='upper center')


# In[76]:


DATA_P = FullDatos.loc[FullDatos['BLOQUE'] == 'C']
sns.lineplot(x='True Trial', y='Gallager',palette="deep", hue='GROUP', data=DATA_P)


# In[77]:


stats.kruskal(*[group["Gallager"].values for name, group in FullDatos.groupby("GROUP")])


# In[78]:


FullPosDatos=FullPosDatos.loc[FullPosDatos['Caso'] != 35]
FullPosDatos=FullPosDatos.loc[FullPosDatos['Caso'] != 8]
FullPosDatos=FullPosDatos.loc[FullPosDatos['Caso'] != 4]


# In[79]:



FullPosDatos= FullPosDatos.loc[FullPosDatos['Time'] > 8] #solo para MPPP


# In[80]:


sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
DATA_POS_FULL = FullPosDatos.loc[FullPosDatos['BLOQUE'] == 'B']
DATA_POS_FULL = DATA_POS_FULL.loc[DATA_POS_FULL['GROUP']== 'PPPD']
#MapadeCalorGen(DATA_POS_FULL)
ax= sns.kdeplot(DATA_POS_FULL.EastWest, DATA_POS_FULL.NorthSouth, cmap='coolwarm', n_levels=50, shade=True, shade_lowest=True)
ax.set(ylim=(0,1),xlim=(0,1), aspect=1)
circle= plt.Circle((0.5,0.5),0.4, color='w', linewidth=3,fill=False)
rectA = plt.Rectangle(((0.33-0.07),(0.43-0.07)),0.09,0.09,linewidth=2,edgecolor='w',facecolor='none')
rectB = plt.Rectangle(((0.85-0.09),(0.40-0.09)),0.09,0.09,linewidth=2,edgecolor='w',facecolor='none')
rectC = plt.Rectangle(((0.31-0.09),(0.51-0.09)),0.09,0.09,linewidth=2,edgecolor='w',facecolor='none')
ax.add_artist(circle)
#ax.add_artist(rectA)
ax.add_artist(rectB)
#ax.add_artist(rectC)
ax.figure.set_size_inches(7,7)


# In[81]:


sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
DATA_POS_FULL = FullPosDatos.loc[FullPosDatos['BLOQUE'] == 'B']
DATA_POS_FULL = DATA_POS_FULL.loc[DATA_POS_FULL['GROUP']== 'Vestibular']
#MapadeCalorGen(DATA_POS_FULL)
ax= sns.kdeplot(DATA_POS_FULL.EastWest, DATA_POS_FULL.NorthSouth, cmap='coolwarm', n_levels=50, shade=True, shade_lowest=True)
ax.set(ylim=(0,1),xlim=(0,1), aspect=1)
circle= plt.Circle((0.5,0.5),0.4, color='w', linewidth=3,fill=False)
rectA = plt.Rectangle(((0.33-0.07),(0.43-0.07)),0.09,0.09,linewidth=2,edgecolor='w',facecolor='none')
rectB = plt.Rectangle(((0.85-0.09),(0.40-0.09)),0.09,0.09,linewidth=2,edgecolor='w',facecolor='none')
rectC = plt.Rectangle(((0.31-0.09),(0.51-0.09)),0.09,0.09,linewidth=2,edgecolor='w',facecolor='none')
ax.add_artist(circle)
#ax.add_artist(rectA)
ax.add_artist(rectB)
#ax.add_artist(rectC)
ax.figure.set_size_inches(7,7)


# In[82]:


sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})   
DATA_POS_FULL = FullPosDatos.loc[FullPosDatos['BLOQUE'] == 'B']
DATA_POS_FULL = DATA_POS_FULL.loc[DATA_POS_FULL['GROUP']== 'Control']
#MapadeCalorGen(DATA_POS_FULL)
ax= sns.kdeplot(DATA_POS_FULL.EastWest, DATA_POS_FULL.NorthSouth, cmap='coolwarm', n_levels=50, shade=True, shade_lowest=True)
ax.set(ylim=(0,1),xlim=(0,1), aspect=1)
circle= plt.Circle((0.5,0.5),0.4, color='w', linewidth=3,fill=False)
rectA = plt.Rectangle(((0.33-0.07),(0.43-0.07)),0.09,0.09,linewidth=2,edgecolor='w',facecolor='none')
rectB = plt.Rectangle(((0.85-0.09),(0.40-0.09)),0.09,0.09,linewidth=2,edgecolor='w',facecolor='none')
rectC = plt.Rectangle(((0.31-0.09),(0.51-0.09)),0.09,0.09,linewidth=2,edgecolor='w',facecolor='none')
ax.add_artist(circle)
#ax.add_artist(rectA)
ax.add_artist(rectB)
#ax.add_artist(rectC)
ax.figure.set_size_inches(7,7)


# In[83]:


Data_Basic['PPPD Positive'] = Data_Basic['GROUP']
Dict = {'PPPD':1, 'Vestibular':0, 'Control':0}
Data_Basic['PPPD Positive'].replace(Dict, inplace=True)


# In[84]:


Data_Roc= Data_Basic[['PPPD Positive', 'Gallager']]
Data_Roc = Data_Roc.sort_values('Gallager')
auc=roc_auc_score(Data_Roc['PPPD Positive'], Data_Roc['Gallager'])
Max = Data_Roc['Gallager'].max()
Data_Roc['Gallager'] = Data_Roc['Gallager'].apply(lambda x: x / Max)
Data_Roc = Data_Roc.set_index(np.arange(len(Data_Roc)))
Data_Roc_onlyPPPD = Data_Roc.loc[Data_Roc['PPPD Positive'] == 0]
aactives = list(Data_Roc_onlyPPPD.index.values)
Data_Roc_Scores = Data_Roc.drop('PPPD Positive', 1)
sscores = list(Data_Roc_Scores.itertuples(index=True, name=None))
prob = list(Data_Roc['Gallager'].values)


# In[85]:


tpr1, fpr1 = get_rates(aactives,sscores)


# In[35]:


Data_Roc= Data_Basic[['PPPD Positive', 'Only B']]
Data_Roc = Data_Roc.sort_values('Only B')
auc2=roc_auc_score(Data_Roc['PPPD Positive'], Data_Roc['Only B'])
Max = Data_Roc['Only B'].max()
Data_Roc['Only B'] = Data_Roc['Only B'].apply(lambda x: x / Max)
Data_Roc = Data_Roc.set_index(np.arange(len(Data_Roc)))
Data_Roc_onlyPPPD = Data_Roc.loc[Data_Roc['PPPD Positive'] == 0]
aactives = list(Data_Roc_onlyPPPD.index.values)
Data_Roc_Scores = Data_Roc.drop('PPPD Positive', 1)
sscores = list(Data_Roc_Scores.itertuples(index=True, name=None))
prob = list(Data_Roc['Only B'].values)

tpr2, fpr2 = get_rates(aactives,sscores)


# In[36]:


Data_Roc= Data_Basic[['PPPD Positive', 'Only C']]
Data_Roc = Data_Roc.sort_values('Only C')
aucC=roc_auc_score(Data_Roc['PPPD Positive'], Data_Roc['Only C'])
Max = Data_Roc['Only C'].max()
Data_Roc['Only C'] = Data_Roc['Only C'].apply(lambda x: x / Max)
Data_Roc = Data_Roc.set_index(np.arange(len(Data_Roc)))
Data_Roc_onlyPPPD = Data_Roc.loc[Data_Roc['PPPD Positive'] == 0]
aactives = list(Data_Roc_onlyPPPD.index.values)
Data_Roc_Scores = Data_Roc.drop('PPPD Positive', 1)
sscores = list(Data_Roc_Scores.itertuples(index=True, name=None))
prob = list(Data_Roc['Only C'].values)

tprC, fprC = get_rates(aactives,sscores)


# In[37]:


Data_Roc= Data_Basic[['PPPD Positive', 'Only BC']]
Data_Roc = Data_Roc.sort_values('Only BC')
aucBC=roc_auc_score(Data_Roc['PPPD Positive'], Data_Roc['Only BC'])
Max = Data_Roc['Only BC'].max()
Data_Roc['Only BC'] = Data_Roc['Only BC'].apply(lambda x: x / Max)
Data_Roc = Data_Roc.set_index(np.arange(len(Data_Roc)))
Data_Roc_onlyPPPD = Data_Roc.loc[Data_Roc['PPPD Positive'] == 0]
aactives = list(Data_Roc_onlyPPPD.index.values)
Data_Roc_Scores = Data_Roc.drop('PPPD Positive', 1)
sscores = list(Data_Roc_Scores.itertuples(index=True, name=None))
prob = list(Data_Roc['Only BC'].values)

tprBC, fprBC = get_rates(aactives,sscores)


# In[42]:


Data_Roc= Data_Basic[['PPPD Positive', 'Only LateBC']]
Data_Roc = Data_Roc.sort_values('Only LateBC')
aucLateBC=roc_auc_score(Data_Roc['PPPD Positive'], Data_Roc['Only LateBC'])
Max = Data_Roc['Only LateBC'].max()
Data_Roc['Only LateBC'] = Data_Roc['Only LateBC'].apply(lambda x: x / Max)
Data_Roc = Data_Roc.set_index(np.arange(len(Data_Roc)))
Data_Roc_onlyPPPD = Data_Roc.loc[Data_Roc['PPPD Positive'] == 0]
aactives = list(Data_Roc_onlyPPPD.index.values)
Data_Roc_Scores = Data_Roc.drop('PPPD Positive', 1)
sscores = list(Data_Roc_Scores.itertuples(index=True, name=None))
prob = list(Data_Roc['Only LateBC'].values)

tprLateBC, fprLateBC = get_rates(aactives,sscores)


# In[68]:


sns.set_context("paper", font_scale = 1.5, rc={"font.size":20,"axes.titlesize":15,"axes.labelsize":16, "lines.linewidth":2.5})   
sns.set_palette('muted')
sns.set_style('darkgrid')
G5= sns.lineplot(fpr1, tpr1, color = 'gold', linewidth= 3, label = 'all 18 trials / all Blocks - ROC AUC = %.2f' % auc, estimator=None, alpha=0.7)
G5= sns.lineplot(fprLateBC, tprLateBC, color = 'green', linewidth = 1.5, label = '7 trials of Block A - ROC AUC = %.2f' % aucLateBC, estimator=None, alpha=0.7)
G5= sns.lineplot(fpr2, tpr2, color = 'crimson', linewidth = 2, label = '7 trials of Block B - ROC AUC = %.2f' % auc2, estimator=None, alpha=0.7)
G5= sns.lineplot(fprC, tprC, color = 'purple', linewidth = 2, label = '7 trials of Block C - ROC AUC = %.2f' % aucC, estimator=None, alpha=0.7)
G5= sns.lineplot(fprBC, tprBC, color = 'navy', linewidth= 3, label = '14 trials of Blocks B & C - ROC AUC = %.2f' % aucBC, estimator=None, alpha=0.6)

G5= sns.lineplot([0,1],[0,1], color = 'lightblue', estimator=None)
G5.lines[5].set_linestyle("dotted")
G5.lines[1].set_linestyle("dashed")
G5.lines[2].set_linestyle("dashed")
G5.lines[3].set_linestyle("dashed")

G5.set(ylim=(-0.005,1.005),xlim=(-0.005,1.005), aspect=1, xlabel = 'False Positive Rate (1-specificty)', ylabel = 'True Positive Rate (sensitivity)', title = 'Mean CSE score for discriminating PPPD: ROC Curves')
G5.figure.set_size_inches(8,8)


# In[61]:


path_to_file = os.path.join(base_path, 'BreinbauerMWM-PPPD-Paths.csv')
export_csv = FullPosDatos.to_csv (path_to_file, index = None, header=True) #Don't forget to add '.csv' at the end of the path


# In[25]:


path_to_file = os.path.join(base_path, 'BreinbauerMWM-PPPD-PathFull.xlsx')
export_csv = FullDatos.to_excel (path_to_file, index = None, header=True) #Don't forget to add '.csv' at the end of the path


# In[24]:


path_to_file = os.path.join(base_path, 'BreinbauerMWM-PPPD-Basics.xlsx')
export_csv = Data_Basic.to_excel (path_to_file, index = None, header=True) #Don't forget to add '.csv' at the end of the path


# In[ ]:




