# ---------------------------------------------------------
# Lab ONCE - Marzo 2023
# Fondecyt 11200469
# Hayo Breinbauer
# ---------------------------------------------------------
#%%

# Felipe, a partir de aqui se "importan" las bibliotecas necesarias para poder hacer cosas en Python"
import pandas as pd     #Bilioteca para manejar Base de datos. Es el equivalente de Excel para Python
import seaborn as sns   #Biblioteca para generar graficos con linda Estetica de gráficos
import matplotlib.pyplot as plt    #Biblioteca para generar Graficos en general
from pathlib import Path # Una sola función dentro de la Bilioteca Path para encontrar archivos en el disco duro

# Aqui vienen una lineas de código solo para encontrar el archivo con la base de datos en el compu:
home= str(Path.home())
Py_Processing_Dir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"
file = Py_Processing_Dir+'AC_PupilLabs_SyncData_Faundez.csv'
# Si tu poner tu archivo .csv en el mismo directorio que tu archivo .py, no hay que hacer tanto jaleo y puede decir:
# file = 'AC_PupilLabs_SyncData_Faundez.csv'

df = pd.read_csv(file, index_col=0) #Aqui cargamos tu base de datos en la variable "df"
file = Py_Processing_Dir+'AB_SimianMaze_Z3_NaviDataBreve_con_calculos.csv'
df2 = pd.read_csv(file, index_col=0)
#%%
df2= df2[df2['Modalidad'] == 'No Inmersivo']

df2.rename(columns={'True_Block': 'MWM_Block', 'True_Trial': 'MWM_Trial'}, inplace=True)

# Perform the merge
df_merged = pd.merge(df, df2[['Sujeto', 'MWM_Block', 'MWM_Trial', 'Duration(ms)', 'Path_length']],
                     on=['Sujeto', 'MWM_Block', 'MWM_Trial'],
                     how='left')
#%%
file=file = Py_Processing_Dir+'Data_para_Faundez_v2.csv'
df_merged.to_csv(file, index=False)
#%%
print('Cargada la base de datos')

# Ahora crearemos una base de datos más pequeña "show_df" donde seleccionaremos solo los datos que nos interesan sin
# perturbar la base de datos general

show_df = df.loc[df['Main_Block']=='Target_is_Hidden'] # Aqui seleccionamos solo los trials con la plataforma invisible
# puedes abrir la base de datos y ver las columnas que hay... cada vez que ponermos [ ]  estamos seleccionando una columna
# o condición.
# con la opción .loc estamos eligiendo solamente las filas que cumplan la condición, en este caso, donde el valor
# de la columna 'Main_Block' sea igual al texto 'Target_is_Hidden'

show_df = show_df.loc[show_df['on_surf']==True] # seguimos sub-seleccionando, ahora solo cuando la mirada esté sobre la superficie
show_df = show_df.reset_index(drop=True) # luego de terminar de seleccionar siempre es importante resetear el indice. Esto es
# un poco más complejo de entender, pero lo importante es hacerlo por si acaso, aunque no siempre es indispensable.


# Y ahora queneramos el gráfico de boxplot que queriamos, evaluando en el Eje Y el valor de la posición en el Eje Y.
sns.set(style= 'white', palette='pastel', font_scale=2,rc={'figure.figsize':(22,12)})
ax= sns.boxplot(data=show_df, x= 'Grupo', y='y_norm')   # Aqui construimos el grafico.
plt.show()  #Y aqui decimos que lo muestre en pantalla.

#LISTO

# Todo lo que viene más abajo es la generación de graficos como tipo mapa de calor. Son lentos, MUUUUUY lentos de generar.

#%%
#ax= sns.kdeplot(data=show_df, x='x_norm', y='y_norm')
#plt.show()
#%%
#ax = sns.kdeplot(
#    data=show_df,
#    x="x_norm",
#    y="y_norm",
#    hue = 'Grupo'
#) #y aqui dibujamos tu grafico.
#ax.set_xlim(0,1)
#ax.set_ylim(0,1)
#plt.show()


print('ready')