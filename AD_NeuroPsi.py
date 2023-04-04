import pandas as pd     #Bilioteca para manejar Base de datos. Es el equivalente de Excel para Python
import seaborn as sns   #Biblioteca para generar graficos con linda Estetica de gráficos
import matplotlib.pyplot as plt    #Biblioteca para generar Graficos en general
from pathlib import Path # Una sola función dentro de la Bilioteca Path para encontrar archivos en el disco duro
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd



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
# Empujando hacia arriba! Banish_List =['P13','P06','P18','P20','P19','P26','P30','P25']
Banish_List =['P13','P30','P34','P02','P07','P15','P10']
#Ver más abajo reporte de eliminados
Banish_List=['P06','P18','P19']
Banish_List=['P06','P26','P36','P17']

b_df = df[df.index.isin(Banish_List)]

df = df[~df.index.isin(Banish_List)]
df = df.loc[df['MOCA']>17]


#df = df.loc[df['Edad']<64]
column_names = df.columns.values.tolist()

df2 = df[['Grupo','Edad']].copy()
print (df2.sort_values(['Grupo', 'Edad'], ascending=[True, True]))
print(df2.groupby(['Grupo']).mean())
print(df2.groupby(['Grupo']).size())
print (df2.sort_values(['Grupo', 'Edad'], ascending=[True, True]))

ax= sns.boxplot(data=df2, x= 'Grupo', y='Edad')   # Aqui construimos el grafico.
plt.show()

df2 = b_df[['Grupo','Edad']].copy()
print('Estos son los eliminados')
print (df2.sort_values(['Grupo', 'Edad'], ascending=[True, True]))
print(df2.groupby(['Grupo']).mean())
print(df2.groupby(['Grupo']).size())
print (df2.sort_values(['Grupo', 'Edad'], ascending=[True, True]))

sns.scatterplot(data=df, x='Edad', y='MOCA', hue='Grupo', s=95)
plt.show()


#sns.set(style= 'white', palette='pastel', font_scale=2,rc={'figure.figsize':(28,12)})

dummy_cols = pd.get_dummies(df['Grupo'], prefix='group')
df3 = pd.concat([df, dummy_cols], axis=1)

for c in column_names:
    if c != 'Grupo':
        X = df3[['Edad', 'group_MPPP', 'group_Vestibular']]
        y = df3[c]
        model = sm.OLS(y, X).fit()

        #print(model.summary())
        df['Var'] = df[c]
        model = ols('Var ~ Grupo + Edad', data=df).fit()
        print ('-------')
        print ('ANALIZANDO = ** ',c)
        print(sm.stats.anova_lm(model, typ=2))
        posthoc = pairwise_tukeyhsd(df[c], df['Grupo'])
        print(posthoc)


        sns.scatterplot(data=df, x='Edad', y=c, hue='Grupo', s=95)
        plt.show()
        sns.boxplot(data=df, x='Grupo', y=c)
        plt.show()
        #sns.barplot(data=df, x='Grupo', y=c, errorbar='se')
        #plt.show()
        #sns.barplot(data=df, x=df.index, y=c, hue='Grupo')
        #plt.show()
        #sns.scatterplot(data=df, x='Edad', y=c, hue='Grupo',s=95)
        #plt.show()
        show_df = df.sort_values(['Grupo', c], ascending=[True, True])

print('Ready')