# ---------------------------------------------------------
# Lab ONCE - Septiembre 2023
# Fondecyt 11200469
# Hayo Breinbauer
# ---------------------------------------------------------

import pandas as pd     #Base de datos
import numpy as np
import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos
from pathlib import Path
import os
import tqdm
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

#--------------------------------------------------------------------------------------------------------------------
#   PREPARAR DIRECTORIOS PARA MANEJO DE DATOS
#--------------------------------------------------------------------------------------------------------------------


home= str(Path.home()) # Obtener el directorio raiz en cada computador distinto
Py_Processing_Dir= home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"
Output_Dir=        home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Outputs/FENRIR v2/"

#--------------------------------------------------------------------------------------------------------------------
#   IMPORTAR BASES DE DATOS
#--------------------------------------------------------------------------------------------------------------------

codex_df = pd.read_csv((Py_Processing_Dir+'AB_SimianMaze_Z4_Resumen_Pacientes_Analizados.csv'), index_col=0)
m_df = pd.read_csv((Py_Processing_Dir+'AB_SimianMaze_Z3_NaviDataBreve_con_calculos.csv'), index_col=0)
p_df = pd.read_csv((Py_Processing_Dir+'AB_SimianMaze_Z2_NaviData_con_posicion.csv'), low_memory=False, index_col=0)
p_df= p_df.reset_index(drop=True)
f_df= pd.read_excel((Py_Processing_Dir+'FENRIR_CODEX.xlsx'), index_col=0)
f_df = f_df.reset_index()
f_df.rename(columns={'CODIGO': 'Sujeto'}, inplace=True)
    # Invocamos en m_df (main Dataframe) la base de datos "corta" con calculo de CSE por Trial
    # Invocamos en p_df (position Dataframe) la base con tutti cuanti - sobre todo datos posicionales

df_CSE = m_df.merge(f_df, on='Sujeto', how='left',suffixes=('', '_f'))
df_Pos = p_df.merge(f_df, on='Sujeto', how='left',suffixes=('', '_f'))
df_Small = f_df

Group_List = ['MPPP', 'Vestibular', 'Voluntario Sano']
Mod_List=['No Inmersivo','Realidad Virtual']
Block_List = ['FreeNav','Training','VisibleTarget_1','VisibleTarget_2','HiddenTarget_1','HiddenTarget_2','HiddenTarget_3']
Nav_List = ['HiddenTarget_1','HiddenTarget_2','HiddenTarget_3']
Subj_List = df_Small['Sujeto'].to_list()
#--------------------------------------------------------------------------------------------------------------------
#   PREPARAR ESTÉTICA EN SEABORN
#--------------------------------------------------------------------------------------------------------------------

sns.set_palette('pastel')
pal = sns.color_palette(n_colors=3)
pal = pal.as_hex()
sns.set(style= 'white', palette='pastel', font_scale=2,rc={'figure.figsize':(12,12)})
Mi_Orden = ['MPPP', 'Vestibular', 'Voluntario Sano']

#--------------------------------------------------------------------------------------------------------------------
#   Gráfico Mapa de Calor
#--------------------------------------------------------------------------------------------------------------------

def MapaDeCalor(dat, Mod, Bloc, Grupo, Titulo):
    colores = sns.color_palette('coolwarm',n_colors=100)
    dat = dat.loc[(dat['Grupo'] == Grupo) & (dat['Modalidad'] == Mod) & (dat['True_Block'] == Bloc)]
    Title = str(Titulo) + str(Mod) + '_' + str(Bloc) + '_' + str(Grupo)
    dat.reset_index()
    sns.set_style("ticks")
    sns.set(style='ticks',rc={"axes.facecolor":colores[0]},font_scale=1.5)
    ax = sns.kdeplot(data=dat, x='P_position_x', y='P_position_y', cmap='coolwarm', n_levels=100, thresh=0, fill=True, cbar=True)

    sns.set_context(font_scale=3)
    ax.set(ylim=(-0.535, 0.535), xlim=(-0.535, 0.535), aspect=1)
    ax.tick_params(labelsize=13)
    ax.set_title(Title, fontsize=22)
    circle = plt.Circle((0, 0), 0.5, color='w',linewidth= 2, fill=False)
    ax.add_artist(circle)
    ax.set(xlabel='East-West (virtual units in Pool-Diameters)', ylabel='North-South (virtual units in Pool-Diameters)')
    plt.xlabel('East-West (virtual units in Pool-Diameters)', fontsize=18)
    plt.ylabel('North-South (virtual units in Pool-Diameters)', fontsize=18)
    ax.figure.set_size_inches(10, 10)
    #plt.grid(False)
    plt.xticks(np.arange(-0.5, 0.75, 0.25))
    plt.yticks(np.arange(-0.5, 0.75, 0.25))

    PSize = (100 / 560)
    rectA = plt.Rectangle(
        (dat['platformPosition_x'].iloc[0] - (PSize / 2), dat['platformPosition_y'].iloc[0] - (PSize / 2)),
        PSize, PSize, linewidth=2.5, edgecolor='yellow',linestyle='--',
        facecolor='none')

    ax.add_artist(rectA)

    directory_path = Output_Dir + 'MapaDeCalor/'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    plt.savefig(directory_path + Title + '.png')

    # plt.show()
    plt.clf()
    print('Mapa de Calor ' + Title + ' Listo')

#--------------------------------------------------------------------------------------------------------------------
#   Grafico Path
#--------------------------------------------------------------------------------------------------------------------

def PathGraph(data, Subj, Mod, Bloc, Titulo):
    if (Bloc != 'FreeNav') and (Bloc !='Training'):
        Plat = True
    else:
        Plat = False

    Title = Titulo + '_' + Subj + '_' + Mod + '_' + Bloc

    show_df = data.loc[(data['Sujeto'] == Subj) & (data['Modalidad'] == Mod) & (data['True_Block'] == Bloc)]
    Grupo = 'NaN'
    if len(show_df) > 0:
        Grupo = show_df['Grupo'].iloc[0]
        unique_count = show_df['True_Trial'].nunique()
        if unique_count > 1:
            ax = sns.lineplot(x="P_position_x", y="P_position_y", hue="True_Trial", data=show_df, linewidth=3, alpha=0.8,
                        legend='full', palette=sns.color_palette('Blues', as_cmap=True), sort=False)  #
        else:
            ax = sns.lineplot(x="P_position_x", y="P_position_y", hue="True_Trial", data=show_df, linewidth=3,
                              alpha=0.8,
                              legend='full', sort=False)  #
        first_datapoints = show_df.groupby('True_Trial')['P_position_x','P_position_y'].first().reset_index()
        for index, row in first_datapoints.iterrows():
            circle = plt.Circle((row['P_position_x'],row['P_position_y']),0.01,color='darkblue',fill=True)
            ax.add_artist(circle)
        sns.set_context("paper", rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 18})
        #sns.set_style('whitegrid')
        circle = plt.Circle((0, 0), 0.5, color='b', fill=False)
        ax.add_artist(circle)
        ax.set(ylim=(-0.535, 0.535), xlim=(-0.535, 0.535),
           aspect=1)  # incluir el equivalente al cuadrado de la pieza completa.
    # el 0.535 considera los 300 pixeles que creo que mide la caja de Simian desde punto centro, divido en los 560 pixeles de Pool Diamater
        ax.set(xlabel='East-West (virtual units in Pool-Diameters)', ylabel='North-South (virtual units in Pool-Diameters)',
           title=Title)
        plt.xlabel('East-West (virtual units in Pool-Diameters)', fontsize=18)
        plt.ylabel('North-South (virtual units in Pool-Diameters)', fontsize=18)
        ax.figure.set_size_inches(7, 7)
        plt.xticks(np.arange(-0.5, 0.75, 0.25))
        plt.yticks(np.arange(-0.5, 0.75, 0.25))
        ax.tick_params(labelsize=13)
        ax.legend(frameon=False, loc='right', bbox_to_anchor=(1.3, 0.5), fontsize=13)
        if Plat:
            PSize = (100 / 560)
            rectA = plt.Rectangle(
                (show_df['platformPosition_x'].iloc[0] - (PSize / 2), show_df['platformPosition_y'].iloc[0] - (PSize / 2)),
                PSize, PSize, linewidth=2, edgecolor='b',
                facecolor='none')

            ax.add_artist(rectA)
        # por Paciente
        directory_path = Output_Dir + 'PathGraph/Por_Sujeto/'+Subj+'/'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        plt.savefig(Output_Dir + 'PathGraph/Por_Sujeto/'+Subj+'/'+Title + '.png')

        # por Bloque
        Title = str(Bloc) + '_' + str(Mod) + '_' + str(Grupo) + '_' + str(Subj)
        directory_path = Output_Dir + 'PathGraph/Por_Bloques/' + Bloc + '/'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        plt.savefig(Output_Dir + 'PathGraph/Por_Bloques/' + Bloc + '/' + Title + '.png')

        #plt.show()
        plt.clf()
        print('Path Graph '+Title+' Listo')

# Resumir CSE por Bloque
CSE_average= df_CSE.groupby(['Sujeto','Modalidad','True_Block'])['CSE'].mean().reset_index()
CSE_average = CSE_average[CSE_average['True_Block'].isin(Nav_List)]
CSE_average = CSE_average.pivot_table(index=['Sujeto'], columns=['Modalidad','True_Block'], values='CSE', aggfunc='first')
CSE_average = CSE_average.reset_index()
df_Small = df_Small.merge(CSE_average, on='Sujeto',how='left',suffixes=('', '_ff') )
CSE_average = df_CSE
CSE_average = CSE_average[CSE_average['True_Block'].isin(Nav_List)]
CSE_average = CSE_average.groupby(['Sujeto','Modalidad'])['CSE'].mean().reset_index()
CSE_average = CSE_average.pivot_table(index=['Sujeto'], columns=['Modalidad'], values='CSE', aggfunc='first')
df_Small = df_Small.merge(CSE_average, on='Sujeto',how='left',suffixes=('', '_fff') )

#CSE_average=
print('Manejo inicial de datos listos - Segmento listo')
#%%
#--------------------------------------------------------------------------------------------------------------------
#   Correr los análisis propiamente tal.
#--------------------------------------------------------------------------------------------------------------------

#%%
# Hacer GraphPaths para todos los sujetos, todos los trials

for index, value in df_Small['Sujeto'].iteritems():
    i=0
    for M in Mod_List:
        for B in Block_List:
            i+=1
            NoGO = False # Para no repetir los GraphPaths..
            if NoGO == False:
                PathGraph(df_Pos, value, M, B, ('FigB_'+str(i)))
print('Todos los Path Graphs, listos')
#--------------------------------------------------------------------------------------------------------------------

#%%
# Revisar los puntos de CSE de algunos trials en particular para identificar outliers.
data = df_CSE
selection = ['VisibleTarget_1','VisibleTarget_2']
data = data[data['True_Block'].isin(selection)]

#ax= sns.scatterplot(data, x='Sujeto', y='CSE', hue = 'True_Trial')
#plt.show()
#--------------------------------------------------------------------------------------------------------------------
#%%
# Hacer un resumen de CSE global por Bloque
for B in Block_List:
    for M in Mod_List:
        for G in Group_List:
            data = df_CSE.loc[(df_CSE['True_Block']==B) & (df_CSE['Grupo']==G) & (df_CSE['Modalidad']==M)]
            Title = B + '_'+M+'_' + G
            ax = sns.barplot(data = data, x='Sujeto', y = 'CSE')
            ax.set(ylim = (0,400),title=Title)
            directory_path = Output_Dir + 'CSE_summary/Por_Bloques/' + B + '/'
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            plt.savefig(directory_path + Title + '.png')
            plt.clf()

print('Ready')

#--------------------------------------------------------------------------------------------------------------------
#%%
# Hacer un resumen de CSE global por Trial!
for B in Block_List:
    for M in Mod_List:
        for S in Subj_List:
            data = df_CSE.loc[(df_CSE['True_Block']==B) & (df_CSE['Sujeto']==S) & (df_CSE['Modalidad']==M)]
            if len(data) > 0:
                Title = S + '_' + M + '_' + B + str(data['Grupo'].iloc[0])

                print(B,M,S,Title)
                ax = sns.barplot(data = data, x='True_Trial', y = 'CSE', hue='Trial_Unique_ID')
                ax.set(ylim = (0,400),title=Title)
                directory_path = Output_Dir + 'CSE_summary/Por_Trial/' + S + '/'
                if not os.path.exists(directory_path):
                   os.makedirs(directory_path)
                plt.savefig(directory_path + Title + '.png')
                plt.clf()

print('We are Ready!!!!')

#--------------------------------------------------------------------------------------------------------------------
#%%
# Heat Maps!
for B in Block_List:
    for M in Mod_List:
        for G in Group_List:
            MapaDeCalor(df_Pos,M,B,G,'T01')
sns.reset_defaults()
sns.set_palette('pastel')
pal = sns.color_palette(n_colors=3)
pal = pal.as_hex()
sns.set(style= 'white', palette='pastel', font_scale=2,rc={'figure.figsize':(12,12)})
print('We are Ready with HeatMaps!!!!')


#--------------------------------------------------------------------------------------------------------------------
#%%
# GRAFICO A!
data = df_CSE[df_CSE['True_Block'].isin(Nav_List)]
data = data[data['Modalidad'].isin(['No Inmersivo'])]

Title = 'A1-CSE por Grupo Global (No Inmersivo)'

ax = sns.boxplot(data, x='Grupo', y='CSE', order=Mi_Orden)
ax.set(ylim=(0, 300), title = Title)
directory_path = Output_Dir + 'Fenir_Outputs/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
plt.savefig(directory_path + Title + '.png')
plt.show()
plt.clf()

data = df_CSE[df_CSE['True_Block'].isin(Nav_List)]

Title = 'A2-CSE por Grupo Global'

ax = sns.boxplot(data, x='Grupo', y='CSE',hue='Modalidad', order=Mi_Orden)
ax.set(ylim=(0, 300), title = Title)
directory_path = Output_Dir + 'Fenir_Outputs/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
plt.savefig(directory_path + Title + '.png')
plt.show()
plt.clf()

#--------------------------------------------------------------------------------------------------------------------
#%%
# GRAFICO B!
data = df_CSE[df_CSE['Modalidad'].isin(['No Inmersivo'])]
data = data[data['True_Block'].isin(Nav_List)]

Title = 'B1-CSE por Bloque por Grupo (No Inmersivo)'
ax = sns.boxplot(data, x='True_Block', y='CSE',hue='Grupo', hue_order=Mi_Orden)
ax.set(ylim=(0, 300), title = Title)
directory_path = Output_Dir + 'Fenir_Outputs/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
plt.savefig(directory_path + Title + '.png')
plt.show()
plt.clf()

data = df_CSE[df_CSE['Modalidad'].isin(['Realidad Virtual'])]
data = data[data['True_Block'].isin(Nav_List)]

Title = 'B2-CSE por Bloque por Grupo (Realidad Virtual)'
ax = sns.boxplot(data, x='True_Block', y='CSE',hue='Grupo', hue_order=Mi_Orden)
ax.set(ylim=(0, 300), title = Title)
directory_path = Output_Dir + 'Fenir_Outputs/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
plt.savefig(directory_path + Title + '.png')
plt.show()
plt.clf()

#--------------------------------------------------------------------------------------------------------------------
#%%
# GRAFICO C!
for M in Mod_List:
    for B in Nav_List:
        for G in Group_List:
            for S in tqdm.tqdm(Subj_List, (B+G),leave=True):
                data = df_CSE[df_CSE['Modalidad'].isin([M])]
                data = data[data['True_Block'].isin([B])]
                data = data[data['Sujeto'].isin([S])]
                data = data[data['Grupo'].isin([G])]
                if len(data)>0:
                    Title = 'D-'+S + ' CSE LearningAng'
                    ax = sns.lineplot(data, x='True_Trial', y='CSE')
                    ax.set(ylim=(0, 300), title=Title)
                    directory_path = Output_Dir + 'Fenir_Outputs/LearningSub/Check_'+M+B+G+'/'
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)
                    plt.savefig(directory_path + Title + '.png')
                    plt.clf()
                    Title = 'D-'+S + ' CSE LearningBAR'
                    ax = sns.barplot(data, x='True_Trial', y='CSE',hue='Trial_Unique_ID')
                    ax.set(ylim=(0, 300), title=Title)
                    directory_path = Output_Dir + 'Fenir_Outputs/LearningSub/Check_'+M+B+G+'/'
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)
                    plt.savefig(directory_path + Title + '.png')
                    plt.clf()

for M in Mod_List:
    for B in tqdm.tqdm(Nav_List,M, leave=True):
        data = df_CSE[df_CSE['Modalidad'].isin([M])]
        data = data[data['True_Block'].isin([B])]

        Title = 'C-'+M+' '+B+' CSE Learning through trial'
        ax = sns.lineplot(data, x='True_Trial', y='CSE', hue='Grupo', hue_order=Mi_Orden)
        ax.set(ylim=(0, 150), title=Title)
        directory_path = Output_Dir + 'Fenir_Outputs/'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        plt.savefig(directory_path + Title + '.png')

        directory_path = Output_Dir + 'Fenir_Outputs/LearningSub/'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        plt.savefig(directory_path + Title + '.png')
        #plt.show()
        plt.clf()

        for G in Group_List:
            data = df_CSE[df_CSE['Modalidad'].isin([M])]
            data = data[data['True_Block'].isin([B])]
            data = data[data['Grupo'].isin([G])]
            Title = 'C2-' + M + ' ' + B + G+ ' CSE Learning'
            ax = sns.lineplot(data, x='True_Trial', y='CSE', hue='Sujeto')
            ax.set(ylim=(0, 150), title=Title)
            directory_path = Output_Dir + 'Fenir_Outputs/'
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            plt.savefig(directory_path + Title + '.png')
            #plt.show()
            plt.clf()
#--------------------------------------------------------------------------------------------------------------------
#%%
# REvision de datos No MWM

sel_cols = df_Small.iloc[:, 9:27]

for column in tqdm.tqdm(sel_cols.columns):
    data=df_Small
    Title = column + '_Box'
    ax = sns.boxplot(data, x='Grupo', y=column, order=Mi_Orden)
    ax.set(title=Title)
    directory_path = Output_Dir + 'Non-MWM/'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    plt.savefig(directory_path + Title + '.png')
    plt.clf()

    for G in Group_List:
        data = df_Small
        data = data[data['Grupo'].isin([G])]
        Title = column + '_'+G+'_Scatter'
        ax = sns.scatterplot(data, x='Sujeto', y=column, s=200)
        for i, row in data.iterrows():
            plt.annotate(row['Sujeto'], (row['Sujeto'], row[column]), textcoords="offset points", xytext=(10, 10), ha='center')

        ax.set(title=Title)
        directory_path = Output_Dir + 'Non-MWM/'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        plt.savefig(directory_path + Title + '.png')
        plt.clf()
print('End of Segmento')

#%%

sel_cols = df_Small.iloc[:, -8:]
for column in tqdm.tqdm(sel_cols.columns):
    data=df_Small
    Title = str(column) + '_Box'
    ax = sns.boxplot(data, x='Grupo', y=column, order=Mi_Orden)
    ax.set(title=Title)
    directory_path = Output_Dir + 'CSEs/'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    plt.savefig(directory_path + Title + '.png')
    plt.clf()

    for G in Group_List:
        data = df_Small
        data = data[data['Grupo'].isin([G])]
        Title = str(column) + '_'+G+'_Scatter'
        ax = sns.scatterplot(data, x='Sujeto', y=column, s=200)
        for i, row in data.iterrows():
            plt.annotate(row['Sujeto'], (row['Sujeto'], row[column]), textcoords="offset points", xytext=(10, 10), ha='center')

        ax.set(title=Title)
        directory_path = Output_Dir + 'CSEs/'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        plt.savefig(directory_path + Title + '.png')
        plt.clf()

#%%

data= df_Small

sel_cols = df_Small.iloc[:, 9:27]
for column in tqdm.tqdm(sel_cols.columns):
    data=df_Small
    Title = 'Cruce ' + column + ' CSE_NI'

    ax = sns.scatterplot(data, x='No Inmersivo', y=column, hue='Grupo', hue_order=Mi_Orden, s=200)
    ax.set(title=Title)
    for i, row in data.iterrows():
        plt.annotate(row['Sujeto'], (row['No Inmersivo'], row[column]), textcoords="offset points", fontsize=9, xytext=(10, 10), ha='center')

    directory_path = Output_Dir + 'Cruces/'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    plt.savefig(directory_path + Title + '.png')
    plt.clf()

#--------------------------------------------------------------------------------------------------------------------
#   End of File
#--------------------------------------------------------------------------------------------------------------------


#%%
df = df_Small
df.drop(df.columns[27:43], axis=1, inplace=True)
corr_matrix = df.corr(method='pearson')

#cse_corr = corr_matrix[['No Inmersivo/HiddenTarget_2']]

plt.figure(figsize=(50, 50))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation with CSE')
plt.show()

#%%
#--------------------------------------------------------------------------------------------------------------------
#   INICIO Graficos finales para Paper 1
#--------------------------------------------------------------------------------------------------------------------

#%%
#--------------------------------------------------------------------------------------------------------------------
#   Edad - Genero -  Nivel educacional
#--------------------------------------------------------------------------------------------------------------------
print('Resumen de edad por grupo')
print (df.sort_values(['Grupo', 'Edad'], ascending=[True, True]))
print(df.groupby(['Grupo']).mean())
print(df.groupby(['Grupo']).size())

df= df_Small

descriptiva = df.groupby('Grupo')['Edad'].describe()
print(descriptiva)

f_val, p_val = stats.f_oneway(df['Edad'][df['Grupo'] == 'MPPP'],
                              df['Edad'][df['Grupo'] == 'Vestibular'],
                              df['Edad'][df['Grupo'] == 'Voluntario Sano'])
print('ANOVA para Edad')
print("Valor F:", f_val)
print("P-valor:", p_val)
print (' ')
print ('Genero')
distribucion_genero = df.groupby(['Grupo', 'Genero']).size().unstack()
distribucion_genero = (distribucion_genero.divide(distribucion_genero.sum(axis=1), axis=0) * 100).round(2)

print(distribucion_genero)

print(' ')
print(' Nivel Educacional ')
descriptiva = df.groupby('Grupo')['N_Educacional'].describe()
print(descriptiva)
f_val, p_val = stats.f_oneway(df['N_Educacional'][df['Grupo'] == 'MPPP'],
                              df['N_Educacional'][df['Grupo'] == 'Vestibular'],
                              df['N_Educacional'][df['Grupo'] == 'Voluntario Sano'])
print('ANOVA para N_Educacional')
print("Valor F:", f_val)
print("P-valor:", p_val)

#%%
#--------------------------------------------------------------------------------------------------------------------
#   Diagnosticos
#--------------------------------------------------------------------------------------------------------------------
print(' ')
df= df_Small
diagnosticos_expandidos = df['Dg'].str.split(',').explode()
diagnosticos_expandidos = diagnosticos_expandidos.apply(lambda x: x.strip() if isinstance(x, str) else x)
# Une los diagnósticos expandidos al DataFrame original para mantener el grupo correspondiente
df_expandido = df.join(diagnosticos_expandidos, rsuffix='_expandido').drop(columns=['Dg'])

# Renombra la columna por claridad
df_expandido = df_expandido.rename(columns={"Dg_expandido": "Diagnostico"})

# Cuenta cada diagnóstico por grupo
conteo_diagnosticos = df_expandido.groupby(['Grupo', 'Diagnostico']).size().unstack(fill_value=0)

# Calcula porcentajes del total de cada grupo
porcentajes_diagnosticos = (conteo_diagnosticos.divide(conteo_diagnosticos.sum(axis=1), axis=0) * 100).round(2)
print (porcentajes_diagnosticos)
#%%
#--------------------------------------------------------------------------------------------------------------------
#   Figura 1
#--------------------------------------------------------------------------------------------------------------------
print(' ')
data = df_CSE[df_CSE['True_Block'].isin(Nav_List)]
data = data[data['Modalidad'].isin(['No Inmersivo'])]

Title = 'Figure 1 - Spatial Navigation Error per Group'

ax = sns.boxplot(data, x='Grupo', y='CSE', linewidth=6, order=Mi_Orden)
ax.set(ylim=(0, 300), title = Title)
directory_path = Output_Dir + 'Paper1_Figures/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
plt.savefig(directory_path + Title + '.png')
plt.show()
plt.clf()

data = df_CSE[df_CSE['True_Block'].isin(Nav_List)]

print('Segmento de script completo - Hayo')
#%%
#--------------------------------------------------------------------------------------------------------------------
#   Figura 2
#--------------------------------------------------------------------------------------------------------------------
print(' ')
data = df_CSE[df_CSE['Modalidad'].isin(['No Inmersivo'])]
data = data[data['True_Block'].isin(Nav_List)]

Title = 'Figure 2 - Spatial Navigation Error per Group at each Experimental Block'
ax = sns.boxplot(data, x='True_Block', y='CSE',hue='Grupo', linewidth=6, hue_order=Mi_Orden)
ax.set(ylim=(0, 300), title = Title)
new_labels = ["Block C", "Block D", "Block E"]
plt.xticks(ticks=range(3), labels=new_labels)
directory_path = Output_Dir + 'Paper1_Figures/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
plt.savefig(directory_path + Title + '.png')
plt.show()
plt.clf()

print('Segmento de script completo - Hayo')
#%%
#--------------------------------------------------------------------------------------------------------------------
#   Figura 3
#--------------------------------------------------------------------------------------------------------------------
print(' ')

data = df_CSE[df_CSE['Modalidad'].isin(['No Inmersivo'])]
#data = data[data['True_Block'].isin([B])]
data = data[data['True_Trial'] != 3]
Title = 'Figure 3 - Spatial Navigation learning through trials'
ax = sns.lineplot(data, x='True_Trial', y='CSE', linewidth = 4, hue='Grupo', hue_order=Mi_Orden)
ax.set(ylim=(0, 150), title=Title)
directory_path = Output_Dir + 'Paper1_Figures/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
plt.savefig(directory_path + Title + '.png')

plt.show()
plt.clf()

print('Segmento de script completo - Hayo')

#%%
#--------------------------------------------------------------------------------------------------------------------
#   Figura 5
#--------------------------------------------------------------------------------------------------------------------
print(' ')
df = df_Small
df['Grupo'] = df['Grupo'].replace({"MPPP":"PPPD", "Vestibular":"Vestibular", "Voluntario Sano":"Control"})
Title= 'Figure 5 - Cognitive tests per group'
df_melted = df.melt(id_vars="Grupo",
                    value_vars=['Niigata', 'DHI','EVA','BDI','STAI_Estado','STAI_Rasgo','MOCA','WAIS_d','WAIS_i','TMT_A_s','TMT_B_s','Corsi_d','Corsi_i','London'],
                    var_name="Variable",
                    value_name="value_column")

g = sns.catplot(
    data=df_melted,
    x='Grupo',
    y='value_column',
    col='Variable',
    col_wrap=3,
    kind='box',
    height=4.5,
    aspect=1.2,
    sharey = False,
    linewidth=4,
    order= ['PPPD','Vestibular','Control']
)
g.set_axis_labels("", "")
custom_labels = {"MPPP":"PPPD", "Vestibular":"Vestibular","Voluntario Sano":"Control"}

g.set_titles("{col_name}")
ax.set(title=Title)
directory_path = Output_Dir + 'Paper1_Figures/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
plt.savefig(directory_path + Title + '.png')

plt.show()
plt.clf()
plt.show()

variables = ['Niigata', 'DHI','EVA','BDI','STAI_Estado','STAI_Rasgo','MOCA','WAIS_d','WAIS_i','TMT_A_s','TMT_B_s','Corsi_d','Corsi_i','London']
df = df.dropna(subset=variables)
for var in variables:
    modelo = ols(f"{var} ~ Grupo", data=df).fit()
    anova = sm.stats.anova_lm(modelo, typ=2)

    print(f"ANOVA para {var}:")
    print(anova)

    # Si el p-valor es menor a 0.05, entonces realizamos pruebas post-hoc
    if anova["PR(>F)"]["Grupo"] < 0.07:
        posthoc = pairwise_tukeyhsd(df[var], df["Grupo"])
        print("\nPruebas post-hoc (Tukey):")
        print(posthoc)

    print("\n" + "-" * 50 + "\n")

print('Segmento de script completo - Hayo')
#%%
#--------------------------------------------------------------------------------------------------------------------
#   Factor Analysys
#--------------------------------------------------------------------------------------------------------------------
print(' ')
df= df_Small
print(df.columns)
cols = ['No Inmersivo', 'Edad','N_Educacional','Edinburgo','Niigata', 'DHI','EVA','BDI','STAI_Estado','STAI_Rasgo','MOCA','WAIS_d','WAIS_i','TMT_A_s','TMT_B_s','Corsi_d','Corsi_i','London']
df= df[cols]
#df.columns = ['_'.join(col).strip() for col in df.columns.values]

df = df.dropna()
# Standardizing the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Applying Factor Analysis
fa = FactorAnalysis(n_components=3,rotation='varimax')  # Here, we're extracting two factors. You can adjust 'n_components' accordingly.
fa_components = fa.fit_transform(df_scaled)

# Loading scores (factor loadings)
loadings = pd.DataFrame(fa.components_, columns=df.columns)
print(loadings.transpose())

# Eigenvalues
eigenvalues = fa.noise_variance_

plt.plot(range(1, df_scaled.shape[1] + 1), eigenvalues, marker='o')
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid(True)
plt.show()



print('Segmento de script completo - Hayo')
#%%

print('Listoco (Script completo) - Hayo')