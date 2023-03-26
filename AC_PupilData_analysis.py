# ---------------------------------------------------------
# Lab ONCE - Enero 2023
# Fondecyt 11200469
# Hayo Breinbauer
# ---------------------------------------------------------
#%%
import pandas as pd     #Base de datos
import numpy as np
import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos
from pathlib import Path

home= str(Path.home())
BaseDir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS/"
file = BaseDir+'AC_PupilLabs_SyncData_Faundez.csv'
df = pd.read_csv(file, index_col=0) #Aqui cargamos tu base de datos
print('Cargado la base de datos')
show_df = df.loc[df['Main_Block']=='Target_is_Hidden'] # Aqui seleccionamos solo los trials con la plataforma invisible
show_df = show_df.loc[show_df['on_surf']==True]
show_df = show_df.reset_index(drop=True)

ax= sns.boxplot(data=show_df, x= 'Grupo', y='y_norm')
plt.show(

#%%
ax= sns.kdeplot(data=show_df, x='x_norm', y='y_norm')
plt.show()
#%%
ax = sns.kdeplot(
    data=show_df,
    x="x_norm",
    y="y_norm",
    hue = 'Grupo'
) #y aqui dibujamos tu grafico.
ax.set_xlim(0,1)
ax.set_ylim(0,1)
plt.show()

show_df = df.loc[df['Main_Block']=='Target_is_Visible'] # Aqui seleccionamos solo los trials con la plataforma invisible
show_df = show_df.loc[show_df['on_surf']==True]
show_df = show_df.reset_index(drop=True)

ax= sns.boxplot(data=show_df, x= 'Grupo', y='y_norm')
plt.show()

ax= sns.boxplot(data=show_df, x= 'Grupo', y='y_norm')
plt.show()

ax = sns.kdeplot(
    data=show_df,
    x="x_norm",
    y="y_norm",
    hue = 'Grupo'
) #y aqui dibujamos tu grafico.
ax.set_xlim(0,1)
ax.set_ylim(0,1)
plt.show()

print('ready')