import pandas as pd     #Bilioteca para manejar Base de datos. Es el equivalente de Excel para Python
import seaborn as sns   #Biblioteca para generar graficos con linda Estetica de gráficos
import matplotlib.pyplot as plt    #Biblioteca para generar Graficos en general
from pathlib import Path # Una sola función dentro de la Bilioteca Path para encontrar archivos en el disco duro

# Aqui vienen una lineas de código solo para encontrar el archivo con la base de datos en el compu:
home= str(Path.home())
Py_Processing_Dir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"
file = Py_Processing_Dir+'AA_CODEX.xlsx'
# Si tu poner tu archivo .csv en el mismo directorio que tu archivo .py, no hay que hacer tanto jaleo y puede decir:
# file = 'AC_PupilLabs_SyncData_Faundez.csv'
df = pd.read_excel(file, index_col=0) #Aqui cargamos tu base de datos en la variable "df"
df = df[['Grupo','Edad']].copy()
Banish_List =['P13','P06','P18','P19','P20','P30']
Banish_List =[]
df = df[~df.index.isin(Banish_List)]
df = df.dropna()
df = df[df.index.notnull()]

print('Cargada la base de datos')
print (df.sort_values(['Grupo', 'Edad'], ascending=[True, True]))
print(df.groupby(['Grupo']).mean())
print(df.groupby(['Grupo']).size())
ax= sns.boxplot(data=df, x= 'Grupo', y='Edad')   # Aqui construimos el grafico.
plt.show()