import pandas as pd     #Bilioteca para manejar Base de datos. Es el equivalente de Excel para Python
import seaborn as sns   #Biblioteca para generar graficos con linda Estetica de gráficos
import matplotlib.pyplot as plt    #Biblioteca para generar Graficos en general
from pathlib import Path # Una sola función dentro de la Bilioteca Path para encontrar archivos en el disco duro

# Aqui vienen una lineas de código solo para encontrar el archivo con la base de datos en el compu:
home= str(Path.home())
Py_Processing_Dir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"
file = Py_Processing_Dir+'AA_NeuroPsi.xlsx'
# Si tu poner tu archivo .csv en el mismo directorio que tu archivo .py, no hay que hacer tanto jaleo y puede decir:
# file = 'AC_PupilLabs_SyncData_Faundez.csv'

df = pd.read_excel(file, index_col=0) #Aqui cargamos tu base de datos en la variable "df"
codex2 = pd.read_excel((Py_Processing_Dir+'AA_CODEX.xlsx'), index_col=0)
Codex_Dict2 = codex2.to_dict('series')
df['Grupo'] = df.index
df['Grupo'].replace(Codex_Dict2['Grupo'], inplace=True)
df['Edad'] = df.index
df['Edad'].replace(Codex_Dict2['Edad'], inplace=True)
print('Cargada la base de datos')
Banish_List =['P13','P06','P18','P30']
df = df[~df.index.isin(Banish_List)]
column_names = df.columns.values.tolist()

df2 = df[['Grupo','Edad']].copy()
print (df2.sort_values(['Grupo', 'Edad'], ascending=[True, True]))
print(df2.groupby(['Grupo']).mean())
print(df2.groupby(['Grupo']).size())
ax= sns.boxplot(data=df2, x= 'Grupo', y='Edad')   # Aqui construimos el grafico.
plt.show()

sns.set(style= 'white', palette='pastel', font_scale=2,rc={'figure.figsize':(28,12)})


for c in column_names:
    if c != 'Grupo':
        sns.boxplot(data=df, x='Grupo', y=c)
        plt.show()
        sns.barplot(data=df, x=df.index, y=c, hue='Grupo')
        plt.show()
        sns.scatterplot(data=df, x='Edad', y=c, hue='Grupo',s=95)
        plt.show()
        print(df.sort_values(['Grupo', c], ascending=[True, True]))

print('Ready')