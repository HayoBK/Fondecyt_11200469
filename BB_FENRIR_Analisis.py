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
from statsmodels.formula.api import mnlogit
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

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

df_Small=  pd.read_excel((Output_Dir + 'Paper1_Figures/df_Loki.xlsx'), index_col=0)

#Corregir nombres de datos vestibulares
new_column_names={
    'vHIT Lat DER GAN':'RL_VOR_Gain',
    'vHIT Lat IZQ GAN':'LL_VOR_Gain',
    'vHIT Ant IZQ GAN':'LA_VOR_Gain',
    'vHIT Post DER GAN':'RP_VOR_Gain',
    'vHIT ANT DER GAN':'RA_VOR_Gain',
    'vHIT Post Izq GAN':'LP_VOR_Gain',
    'vHIT LAT DER':'RL_vHIT_Saccade',
    'vHIT LAT IZQ':'LL_vHIT_Saccade',
    'vHIT ANT IZQ':'LA_vHIT_Saccade',
    'vHIT POST DER':'RP_vHIT_Saccade',
    'vHIT ANT DER':'RA_vHIT_Saccade',
    'vHIT POST IZQ':'LP_vHIT_Saccade',
    'oVEMP DER':'R_oVEMP',
    'oVEMP IZQ':'L_oVEMP',
    'cVEMP DER':'R_cVEMP',
    'cVEMP IZQ':'L_cVEMP'
}
def Vemp_Ajuste(df, col_idx1, col_idx2, threshold):
    for index, row in df.iterrows():
        while row[df.columns[col_idx1]] >= threshold or row[df.columns[col_idx2]] >= threshold:
            row[df.columns[col_idx1]] /= 10
            row[df.columns[col_idx2]] /= 10
        df.at[index, df.columns[col_idx1]] = row[df.columns[col_idx1]]
        df.at[index, df.columns[col_idx2]] = row[df.columns[col_idx2]]
    return df
"""
df_Small = Vemp_Ajuste(df_Small, 39, 40, 0.5)
df_Small = Vemp_Ajuste(df_Small, 41, 42, 0.5)
df_Small[df_Small.columns[39]] *= 100
df_Small[df_Small.columns[40]] *= 100
df_Small[df_Small.columns[41]] *= 1000
df_Small[df_Small.columns[42]] *= 1000
"""
df_Small.rename(columns=new_column_names, inplace=True)

#Procesar datos vestibulares
df_Small['VOR_6Gain'] = df_Small[['RL_VOR_Gain','LL_VOR_Gain','LA_VOR_Gain','RP_VOR_Gain','RA_VOR_Gain','LP_VOR_Gain']].mean(axis=1)
df_Small['VOR_R3Gain'] = df_Small[['RL_VOR_Gain','RP_VOR_Gain','RA_VOR_Gain']].mean(axis=1)
df_Small['VOR_L3Gain'] = df_Small[['LL_VOR_Gain','LA_VOR_Gain','LP_VOR_Gain']].mean(axis=1)
df_Small['VEMP_4Gain'] = df_Small[['R_oVEMP','L_oVEMP','L_cVEMP','R_cVEMP']].mean(axis=1)
df_Small['Best_VOR_Lateral'] = np.where(df_Small['RL_VOR_Gain'] > df_Small['LL_VOR_Gain'],df_Small['RL_VOR_Gain'], df_Small['LL_VOR_Gain'])
df_Small['Worse_VOR_Lateral'] = np.where(df_Small['RL_VOR_Gain'] > df_Small['LL_VOR_Gain'],df_Small['LL_VOR_Gain'], df_Small['RL_VOR_Gain'])
df_Small['Best_VOR_Anterior'] = np.where(df_Small['RA_VOR_Gain'] > df_Small['LA_VOR_Gain'],df_Small['RA_VOR_Gain'], df_Small['LA_VOR_Gain'])
df_Small['Worse_VOR_Anterior'] = np.where(df_Small['RA_VOR_Gain'] > df_Small['LA_VOR_Gain'],df_Small['LA_VOR_Gain'], df_Small['RA_VOR_Gain'])
df_Small['Best_VOR_Posterior'] = np.where(df_Small['RP_VOR_Gain'] > df_Small['LP_VOR_Gain'],df_Small['RP_VOR_Gain'], df_Small['LP_VOR_Gain'])
df_Small['Worse_VOR_Posterior'] = np.where(df_Small['RP_VOR_Gain'] > df_Small['LP_VOR_Gain'],df_Small['LP_VOR_Gain'], df_Small['RP_VOR_Gain'])
df_Small['Best_oVEMP'] = np.where(df_Small['R_oVEMP'] > df_Small['L_oVEMP'],df_Small['R_oVEMP'], df_Small['L_oVEMP'])
df_Small['Worse_oVEMP'] = np.where(df_Small['R_oVEMP'] > df_Small['L_oVEMP'],df_Small['L_oVEMP'], df_Small['R_oVEMP'])
df_Small['Best_cVEMP'] = np.where(df_Small['R_cVEMP'] > df_Small['L_cVEMP'],df_Small['R_cVEMP'], df_Small['L_cVEMP'])
df_Small['Worse_cVEMP'] = np.where(df_Small['R_cVEMP'] > df_Small['L_cVEMP'],df_Small['L_cVEMP'], df_Small['R_cVEMP'])
df_Small['Saccades6'] = df_Small.iloc[:, [33, 38]].sum(axis=1)
df_Small['Best_Sac_Lateral'] = np.where(df_Small['RL_VOR_Gain'] > df_Small['LL_VOR_Gain'],df_Small['RL_vHIT_Saccade'], df_Small['LL_vHIT_Saccade'])
df_Small['Worse_Sac_Lateral'] = np.where(df_Small['RL_VOR_Gain'] > df_Small['LL_VOR_Gain'],df_Small['LL_vHIT_Saccade'], df_Small['RL_vHIT_Saccade'])
df_Small['Best_Sac_Anterior'] = np.where(df_Small['RA_VOR_Gain'] > df_Small['LA_VOR_Gain'],df_Small['RA_vHIT_Saccade'], df_Small['LA_vHIT_Saccade'])
df_Small['Worse_Sac_Anterior'] = np.where(df_Small['RA_VOR_Gain'] > df_Small['LA_VOR_Gain'],df_Small['LA_vHIT_Saccade'], df_Small['RA_vHIT_Saccade'])
df_Small['Best_Sac_Posterior'] = np.where(df_Small['RP_VOR_Gain'] > df_Small['LP_VOR_Gain'],df_Small['RP_vHIT_Saccade'], df_Small['LP_vHIT_Saccade'])
df_Small['Worse_Sac_Posterior'] = np.where(df_Small['RP_VOR_Gain'] > df_Small['LP_VOR_Gain'],df_Small['LP_vHIT_Saccade'], df_Small['RP_vHIT_Saccade'])
print('--------------------------')
print('PROCESAMIENTO BÁSICO LISTO')
print('--------------------------')


#%%
results = []
List= list(range(27,43))
column_List=List
List= list(range(53,67))
column_List= list(range(27,43)) + list(range(53,67))
# Iterate over each vestibular function column
extra=[6,9,67,68,69,70,71,72,73]
column_List = column_List + extra
sac = list(range(33, 39))
g= [1]
sac = sac + extra +g
filtered_df = df_Small[df_Small['Grupo'].isin(['Voluntario Sano', 'MPPP'])]
vestlist = column_List + sac
filtered_dfB = df_Small.iloc[:, vestlist]
grouped = filtered_dfB.groupby('Grupo').agg(['mean','std'])
grouped.to_excel((Output_Dir+'Vestibulares.xlsx'))
print('--------Go-------------')
for column in df_Small.drop('Grupo', axis=1).columns:
    print('--------------------')
    print(column)
    print(df_Small.columns.get_loc(column))

    # Calculate mean and standard deviation for each group
    if (df_Small.columns.get_loc(column) in sac):
        Porcentaje_Sacadas_enGrupo = df_Small.groupby('Grupo')[column].mean() * 100
        print('')
        print(Porcentaje_Sacadas_enGrupo)

    if pd.api.types.is_numeric_dtype(df_Small[column]) and (df_Small.columns.get_loc(column) in column_List) :
        print('Es numérico, asi que demosle...')
        print(' ')
        group_stats = df_Small.groupby('Grupo')[column].agg(['mean', 'std'])
        # Perform ANOVA
        fvalue, pvalue = stats.f_oneway(df_Small[df_Small['Grupo'] == 'MPPP'][column],
                                    df_Small[df_Small['Grupo'] == 'Vestibular'][column],
                                    df_Small[df_Small['Grupo'] == 'Voluntario Sano'][column])

        # If ANOVA is significant, perform post-hoc test
        t_statistic, tp_value = stats.ttest_ind(
            filtered_df[filtered_df['Grupo'] == 'MPPP'][column],
            filtered_df[filtered_df['Grupo'] == 'Voluntario Sano'][column],
            equal_var=False  # Assumes unequal variance
        )
        if pvalue < 0.05:
            mc = pairwise_tukeyhsd(df_Small[column], df_Small['Grupo'])
            posthoc_result = mc.summary()
        else:
            posthoc_result = "No significant difference found"

        # Append results
        results.append({
            'Function': column,
            'Group Statistics': group_stats,
            'ANOVA F-value': fvalue,
            'ANOVA p-value': pvalue,
            'Post-hoc Test Result': posthoc_result
        })
        print('Function:            ', column)
        print('Group Statistics:    ', group_stats)
        print('MPPP-Sano T    ',t_statistic)
        print('MPPP-sano T    ',tp_value)
        if tp_value < 0.05:
            print('----****-----')
            print('    SIG      ')
            print('----***------')
        print('ANOVA F-value:       ', fvalue)
        print('ANOVA p-value:       ', pvalue)
        print('Post-hoc Test Result:', posthoc_result)

        ax = sns.boxplot(df_Small, x='Grupo', y=column, linewidth=6, order=Mi_Orden)
        Title = column
        directory_path = Output_Dir + 'Vestibulars/'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        plt.savefig(directory_path + Title + '.png')
        plt.clf()

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)
#print(results_df)

print('Listo con calculos de función vestibular')

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

data= df_Small
groups = [data['No Inmersivo'][data['Grupo'] == g] for g in data['Grupo'].unique()]
f_val, p_val = stats.f_oneway(*groups)

print("ANOVA results:")
print("F-value:", f_val)
print("P-value:", p_val)

posthoc = pairwise_tukeyhsd(data['No Inmersivo'], data['Grupo'], alpha=0.05)
print(posthoc)

Title = 'Figure 1 - Spatial Navigation Error per Group'

#Summarized
data=df_Small

ax = sns.boxplot(data, x='Grupo', y='No Inmersivo', linewidth=6, order=Mi_Orden)
sns.stripplot(data=data, x='Grupo', y='No Inmersivo', jitter=True, color='black', size=10, ax=ax, order=Mi_Orden)

ax.set_ylabel("Cummulative search error (CSE) in pool diameters", weight='bold')
ax.set_xlabel("Group", weight='bold')
ax.set_xticklabels(["PPPD", "Vestibular", "Healthy control"])

ax.set(ylim=(0, 180))
ax.set_title(Title, weight='bold')
# Determine the y position for the line and annotation
y_max = 142 + 10  # 10 units above the highest data point; adjust as needed

# Draw lines for PPPD vs Vestibular
plt.plot([0, 0, 0.95, 0.95], [y_max, y_max + 5, y_max + 5, y_max], lw=4.5, color='black')
plt.text(0.5, y_max - 0, "*", ha='center', va='bottom', color='black', weight='bold', fontsize=40)

y_max -= 3
plt.plot([1.05, 1.05, 2, 2], [y_max, y_max + 5, y_max + 5, y_max], lw=4.5, color='black')
plt.text(1.5, y_max - 0, "*", ha='center', va='bottom', color='black', weight='bold', fontsize=40)

y_max +=3
# For PPPD vs Healthy control, we adjust the y-values to place the annotation a bit higher
y_max += 8  # Adjusting for the next significant difference bar; tweak as necessary
plt.plot([0, 0, 2, 2], [y_max, y_max + 5, y_max + 5, y_max], lw=4.5, color='black')
plt.text(1, y_max - 0, "*", ha='center', va='bottom', color='black', weight='bold', fontsize=40)


directory_path = Output_Dir + 'Paper1_Figures/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
plt.savefig(directory_path + Title + '.png')
plt.show()
plt.clf()

data = df_CSE[df_CSE['True_Block'].isin(Nav_List)]

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

blocks = data['True_Block'].unique()

for block in blocks:
    block_data = data[data['True_Block'] == block]

    # One-way ANOVA
    groups = [block_data['CSE'][block_data['Grupo'] == g] for g in block_data['Grupo'].unique()]
    f_val, p_val = stats.f_oneway(*groups)
    print(f"\nANOVA results for {block}:")
    print("F-value:", f_val)
    print("P-value:", p_val)

    # Tukey's post-hoc test
    if p_val < 0.05:  # Check significance level; adjust if needed
        posthoc = pairwise_tukeyhsd(block_data['CSE'], block_data['Grupo'])
        print("Post-hoc results:")
        print(posthoc)

print('Segmento de script completo - Hayo')
#%%
#--------------------------------------------------------------------------------------------------------------------
#   Figura 2
#--------------------------------------------------------------------------------------------------------------------
print(' ')
data = df_CSE[df_CSE['Modalidad'].isin(['No Inmersivo'])]
This_List=[]
This_List = Nav_List
This_List.extend(['Training', 'VisibleTarget_1','VisibleTarget_2'])
data = data[data['True_Block'].isin(This_List)]
data = data.groupby(['Sujeto', 'Grupo', 'Modalidad', 'True_Block'])['CSE'].mean()
indices_to_drop =[0,239]
data = data.drop(indices_to_drop)
data.reset_index(drop=True, inplace=True)
data = data.reset_index()
Title = 'Figure 2 - Spatial Navigation Error per Group at each Experimental Block'
true_block_order = ['Training', 'VisibleTarget_1', 'HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3', 'VisibleTarget_2']
ax = sns.boxplot(data, x='True_Block', y='CSE',hue='Grupo', linewidth=6,order=true_block_order, hue_order=Mi_Orden)
ax.set_ylabel("Cummulative search error (CSE) in pool diameters", weight='bold', fontsize=19)

ax.set(ylim=(0, 300))
ax.set_title(Title, weight='bold', fontsize=20)
ax.set_xlabel("")
new_labels = ["Block A\nTraining", "Block B\nTarget \nVisible", "Block C\nTarget\nHidden","Block D\nTarget\nHidden","Block E\nTarget\nHidden\nRandom \nstarting\npoint","Block F\nTarget \nVisible"]
plt.xticks(ticks=range(6), labels=new_labels)

handles, labels = ax.get_legend_handles_labels()
new_labels = ["PPPD", "Vestibular", "Healthy control"]
ax.legend(handles, new_labels, title="Group", loc='upper left',prop={'size': 18}, title_fontsize=20)

y_max=275
plt.plot([1.72, 1.72, 2.23, 2.23], [y_max, y_max + 5, y_max + 5, y_max], lw=3.5, color='black')
plt.text(2, y_max - 2, "*", ha='center', va='bottom', color='black', weight='bold', fontsize=30)

y_max=265
plt.plot([1.72, 1.72, 2, 2], [y_max, y_max + 5, y_max + 5, y_max], lw=3.5, color='black')
plt.text(1.875, y_max - 2, "*", ha='center', va='bottom', color='black', weight='bold', fontsize=30)

y_max=275
plt.plot([2.72, 2.72, 3.23, 3.23], [y_max, y_max + 5, y_max + 5, y_max], lw=3.5, color='black')
plt.text(3, y_max - 2, "*", ha='center', va='bottom', color='black', weight='bold', fontsize=30)

y_max=265
plt.plot([2.72, 2.72, 3, 3], [y_max, y_max + 5, y_max + 5, y_max], lw=3.5, color='black')
plt.text(2.875, y_max - 2, "*", ha='center', va='bottom', color='black', weight='bold', fontsize=30)

y_max=285
plt.plot([3.72, 3.72, 4.23, 4.23], [y_max, y_max + 5, y_max + 5, y_max], lw=3.5, color='black')
plt.text(4, y_max - 2, "*", ha='center', va='bottom', color='black', weight='bold', fontsize=30)

y_max=275
plt.plot([3.72, 3.72, 4, 4], [y_max, y_max + 5, y_max + 5, y_max], lw=3.5, color='black')
plt.text(3.875, y_max - 2, "*", ha='center', va='bottom', color='black', weight='bold', fontsize=30)

def cohens_d(group1, group2):
    pooled_std = np.sqrt((np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2)
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d

def bootstrap_difference(data1, data2, num_samples=10000):
    diffs = []
    for _ in range(num_samples):
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)
        diffs.append(cohens_d(sample1, sample2))
    return diffs


directory_path = Output_Dir + 'Paper1_Figures/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
plt.tight_layout()
plt.savefig(directory_path + Title + '.png')
plt.show()
plt.clf()

#%%
blocks = ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']  # Adjust block names as per your data

for block in blocks:
    block_data = data[data['True_Block'] == block]
    groups = [block_data['CSE'][block_data['Grupo'] == g].values for g in Mi_Orden]

    # Calculate effect size and bootstrap difference for each pair of groups
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            print(f"\nComparing {Mi_Orden[i]} vs {Mi_Orden[j]} in {block}:")

            d = cohens_d(groups[i], groups[j])
            print(f"Cohen's d: {d}")

            diffs = bootstrap_difference(groups[i], groups[j])
            plt.hist(diffs, bins=50, alpha=0.5, label=f"{Mi_Orden[i]} vs {Mi_Orden[j]}")

    plt.title(f"Bootstrapped Differences in {block}")
    plt.xlabel("Effect Size (Cohen's d)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
blocks = ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']  # Adjust block names as per your data

for block in blocks:
    block_data = data[data['True_Block'] == block]
    groups = [block_data['CSE'][block_data['Grupo'] == g].values for g in Mi_Orden]

    # Calculate Levene's test for homogeneity of variance
    stat, p = stats.levene(*groups)

    print(f"\nHomogeneity of Variance Test for {block}:")
    print(f"Statistic: {stat}")
    print(f"P-value: {p}")

    if p > 0.05:
        print(f"Result: Homogeneous variances (p > 0.05)")
    else:
        print(f"Result: Heterogeneous variances (p <= 0.05)")

    # Calculate effect size and bootstrap difference for each pair of groups (as before)
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            print(f"\nComparing {Mi_Orden[i]} vs {Mi_Orden[j]} in {block}:")

            d = cohens_d(groups[i], groups[j])
            print(f"Cohen's d: {d}")

            diffs = bootstrap_difference(groups[i], groups[j])
            plt.hist(diffs, bins=50, alpha=0.5, label=f"{Mi_Orden[i]} vs {Mi_Orden[j]}")

    plt.title(f"Bootstrapped Differences in {block}")
    plt.xlabel("Effect Size (Cohen's d)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
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
line_styles_dict = {
    Mi_Orden[0]: '-',      # Solid line for first group
    Mi_Orden[1]: '--',     # Dashed line for second group
    Mi_Orden[2]: '-.'      # Dash-dot line for third group
}
ax = sns.lineplot(data, x='True_Trial', y='CSE', linewidth = 4, hue='Grupo', hue_order=Mi_Orden)
ax.set(ylim=(0, 150))

ax.set_title(Title, weight='bold', fontsize=24)
ax.set_ylabel("Cummulative search error (CSE) in pool diameters", weight='bold', fontsize=22)
handles, labels = ax.get_legend_handles_labels()
new_labels = ["PPPD", "Vestibular", "Healthy control"]
#handles = handles[1:]
for handle in handles:  # Skip the title
    handle.set_linewidth(10)
ax.legend(handles, new_labels, title="Group", loc='upper left',prop={'size': 20}, title_fontsize=22)
ax.set_xlabel("Trial", weight='bold', fontsize=22)


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
df = df[df['Sujeto'] != 'P13']
df['Grupo'] = df['Grupo'].replace({"MPPP":"PPPD", "Vestibular":"Vestibular", "Voluntario Sano":"Control"})
Title= 'Figure 5 - Cognitive tests per group'
df_melted = df.melt(id_vars="Grupo",
                    value_vars=['Niigata', 'DHI','EVA','BDI','STAI_Estado','STAI_Rasgo','MOCA','WAIS_d','WAIS_i','TMT_A_s','TMT_B_s','Corsi_d','Corsi_i','London'],
                    var_name="Variable",
                    value_name="value_column")
df_melted.loc[df_melted['Variable'] == 'MOCA', 'value_column'] *= -1
df_melted.loc[df_melted['Variable'] == 'WAIS_d', 'value_column'] *= -1
df_melted.loc[df_melted['Variable'] == 'WAIS_i', 'value_column'] *= -1
df_melted.loc[df_melted['Variable'] == 'Corsi_d', 'value_column'] *= -1
df_melted.loc[df_melted['Variable'] == 'Corsi_i', 'value_column'] *= -1
df_melted.loc[df_melted['Variable'] == 'London', 'value_column'] *= -1

new_titles_dict = {
    'Niigata':'Niigata - PPPD symptoms',
    'DHI':'DHI - Dizziness impact on life',
    'EVA':'AVSD - Dizziness intensity',
    'BDI':'BDI - Depressive symtpoms',
    'STAI_Estado':'STAI - State Anxiety',
    'STAI_Rasgo':'STAI - Trait Anxiety',
    'MOCA':'MoCA (-1*)- Global cognition',
    'WAIS_d':'DST (-1*)- digit span memory',
    'WAIS_i':'DST(inverted) (-1*)- digit memory',
    'TMT_A_s':'TMT A - visuospatial attention ',
    'TMT_B_s':'TMT B - visuospatial executive function',
    'Corsi_d':'CBTT (-1*)- visuospatial memory',
    'Corsi_i':'CBTT(inverted) (-1*)- visuospatial memory',
    'London':'ToL (-1*)- Visuospatial planning'
}
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
    order= ['PPPD','Vestibular','Control'],
    legend= False
)
g.set_axis_labels("", "")
custom_labels = {"MPPP":"PPPD", "Vestibular":"Vestibular","Voluntario Sano":"Control"}

g.set_titles("{col_name}")
for ax, title in zip(g.axes.flat, g.col_names):
    new_title = new_titles_dict.get(title, title)  # Fetch new title or default to original
    ax.set_title(new_title, fontsize=19, weight='bold')
# .
# .
#ax.set(title=Title)
directory_path = Output_Dir + 'Paper1_Figures/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
plt.savefig(directory_path + Title + '.png')

plt.show()
plt.clf()
plt.show()

variables = ['Niigata', 'DHI','EVA','BDI','STAI_Estado','STAI_Rasgo','MOCA','WAIS_d','WAIS_i','TMT_A_e','TMT_B_e','TMT_A_s','TMT_B_s','Corsi_d','Corsi_i','London']
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
df = df.rename(columns={"No Inmersivo": "CSE"})
df = df[df['Sujeto'] != 'P13']
print(df.columns)
cols = ['CSE', 'Edad','N_Educacional','Edinburgo','Niigata', 'DHI','EVA','BDI','STAI_Estado','STAI_Rasgo','MOCA','WAIS_d','WAIS_i','TMT_A_s','TMT_B_s','Corsi_d','Corsi_i','London']
df= df[cols]

#df.columns = ['_'.join(col).strip() for col in df.columns.values]

#df = df.dropna()
df.fillna(df.mean(), inplace=True)
# Standardizing the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Applying Factor Analysis
fa = FactorAnalysis(n_components=3, rotation = 'varimax')  # Here, we're extracting two factors. You can adjust 'n_components' accordingly.
fa_components = fa.fit_transform(df_scaled)

# Loading scores (factor loadings)
loadings = pd.DataFrame(fa.components_, columns=df.columns)
loadings = loadings.transpose()
print(loadings)

data = {
    'Factor 1': {
        "CSE": -0.316401,

        "Niigata": -0.723562,
        "DHI": -0.931750,
        "EVA": -0.921200,
        "BDI": -0.411383,
        "STAI_Estado": -0.120432,
        "STAI_Rasgo": -0.445399,
        "MOCA": -0.193202,
        "WAIS_d": -0.213498,
        "WAIS_i": -0.217129,
        "TMT_A_s": -0.212810,
        "TMT_B_s": -0.076990,
        "Corsi_d": -0.230282,
        "Corsi_i": -0.250927,
        "London": -0.114779,
        "Edad": -0.074150,
        "N_Educacional": 0.157497
    },
    'Factor 2': {
        "CSE": 0.577380,

        "Niigata": 0.174412,
        "DHI": 0.160030,
        "EVA": 0.070375,
        "BDI": 0.004473,
        "STAI_Estado": 0.116069,
        "STAI_Rasgo": 0.105681,
        "MOCA": 0.502769,
        "WAIS_d": 0.085023,
        "WAIS_i": -0.045009,
        "TMT_A_s": 0.747984,
        "TMT_B_s": 0.616490,
        "Corsi_d": 0.242118,
        "Corsi_i": 0.440112,
        "London": 0.299767,
        "Edad": 0.767359,
        "N_Educacional": -0.367497
    },
    'Factor 3': {
        "CSE": -0.252049,

        "Niigata": -0.292546,
        "DHI": -0.097016,
        "EVA": -0.104390,
        "BDI": -0.191333,
        "STAI_Estado": 0.103416,
        "STAI_Rasgo": -0.121329,
        "MOCA": -0.306277,
        "WAIS_d": -0.453122,
        "WAIS_i": -0.825399,
        "TMT_A_s": -0.036014,
        "TMT_B_s": -0.444738,
        "Corsi_d": -0.423343,
        "Corsi_i": -0.386639,
        "London": -0.548620,
        "Edad": 0.190805,
        "N_Educacional": 0.127497

    }
}

new_titles_dict = {
    'Niigata':'Niigata - PPPD symptoms',
    'DHI':'DHI - Dizziness impact on life',
    'EVA':'AVSD - Dizziness intensity',
    'CSE':'CSE - Spatial navigation error',
    'BDI':'BDI - Depressive symtpoms',
    'STAI_Estado':'STAI - State Anxiety',
    'STAI_Rasgo':'STAI - Trait Anxiety',
    'MOCA':'MoCA (-1*)- Global cognition',
    'WAIS_d':'DST (-1*)- digit span memory',
    'WAIS_i':'DST(inverted) (-1*)- digit memory',
    'TMT_A_s':'TMT A - visuospatial attention ',
    'TMT_B_s':'TMT B - visuospatial executive function',
    'Corsi_d':'CBTT (-1*)- visuospatial memory',
    'Corsi_i':'CBTT(inverted) (-1*)- visuospatial memory',
    'London':'ToL (-1*)- Visuospatial planning',
    'Edad':'Age',
    'N_Educacional':'Educational level'

}

loadings = pd.DataFrame(data)
loadings = loadings.rename(index=new_titles_dict)


# Heatmap visualization
plt.figure(figsize=(15, 12))
sns.heatmap(loadings, annot=True, cmap="coolwarm", center=0, linewidths=.5, linecolor='black', fmt=".2f")
plt.title('Figure 7 - Factor Loadings Heatmap', fontsize=28, weight='bold')
plt.xlabel('Factors', fontsize= 26, weight= 'bold')
plt.xticks(fontsize=22, weight='bold')

plt.tight_layout()
Title = 'Figure 7 - Factor Loadings Heatmap'
directory_path = Output_Dir + 'Paper1_Figures/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
plt.savefig(directory_path + Title + '.png')
plt.show()
plt.clf()

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
#--------------------------------------------------------------------------------------------------------------------
#   Regresiones
#--------------------------------------------------------------------------------------------------------------------
print(' ')
df= df_Small
df = df[df['Sujeto'] != 'P13']
cols = ['Grupo','No Inmersivo', 'Edad','N_Educacional','Edinburgo','Niigata', 'DHI','EVA','BDI','STAI_Estado','STAI_Rasgo','MOCA','WAIS_d','WAIS_i','TMT_A_s','TMT_B_s','Corsi_d','Corsi_i','London']
df= df[cols]
df['Grupo'] = pd.Categorical(df['Grupo'])
#df.fillna(df.mean(), inplace=True)
df = df.dropna()
grupo_dummies = pd.get_dummies(df['Grupo'], prefix='Grupo', drop_first=False)

# Add the dummy columns back to the dataframe
df = pd.concat([df, grupo_dummies], axis=1)


# Define dependent and independent variables
X = df[['No Inmersivo', 'Edad','N_Educacional','Edinburgo','Niigata', 'DHI','EVA','BDI','STAI_Estado','STAI_Rasgo','MOCA','WAIS_d','WAIS_i','TMT_A_s','TMT_B_s','Corsi_d','Corsi_i','London']]
X = sm.add_constant(X)  # Adds a constant column for the intercept
y = df['Niigata']

# Create the model
model = sm.OLS(y, X)

# Fit the model
results = model.fit()

# Print out the statistics
print(results.summary())

# Setting up the logistic regression model
formula = "Grupo_MPPP ~ Edad + N_Educacional  +  BDI + STAI_Estado + STAI_Rasgo + MOCA + WAIS_i + TMT_A_s + TMT_B_s + Corsi_i + London"  # ... add all other independent variables
model = mnlogit(formula, data=df).fit()

# Print the summary
print(model.summary())

print('Segmento de script completo - Hayo')


#%%
#--------------------------------------------------------------------------------------------------------------------
#   Figura 7 Correlaciones.
#--------------------------------------------------------------------------------------------------------------------
print(' ')
data= df_Small
data = data.rename(columns={"No Inmersivo": "CSE"})

Title = 'Figure 7 - PPPD symptoms and Spatial Navigation Impairment'

ax = sns.scatterplot(data, x='Niigata', y='CSE', style='Grupo' , hue='Grupo', palette="deep", hue_order=Mi_Orden, s=100 )
ax.set(title=Title)
plt.xlabel('CSE - Spatial Navigation error')
directory_path = Output_Dir + 'Paper1_Figures/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
plt.savefig(directory_path + Title + '.png')
plt.show()
plt.clf()

Title = 'Figure 7 - PPPD symptoms and Spatial Navigation Impairment'

# Create lmplot
g = sns.lmplot(data=data, x='Niigata', y='CSE', markers=['o','s','x'], hue='Grupo', palette="deep", hue_order=Mi_Orden,
               height=8, aspect=1, ci=None, scatter_kws={'s': 75}, line_kws={'linewidth': 4.5})
sns.regplot(data=data, x='Niigata', y='CSE', scatter=False,ci=None, ax=g.axes[0, 0], color='grey', line_kws={'linestyle': '--', 'linewidth': 3.2})

# Set titles and labels
plt.subplots_adjust(top=0.8)
g.fig.suptitle(Title, y= 0.96, weight='bold', fontsize=22)
g.set_xlabels('Niigata - PPPD symptoms level', fontsize=20, weight='bold')
g.set_ylabels("Cummulative search error (CSE) in pool diameters", fontsize=19, weight='bold')

# Customize the legend
new_labels = ["PPPD", "Vestibular", "Healthy control"]
g._legend.set_title("Group")
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)

# If you still need this line (for x-tick labels) uncomment it, but it seems unrelated to this plot:
# plt.xticks(ticks=range(6), labels=new_labels)

# Save the plot
directory_path = Output_Dir + 'Paper1_Figures/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
plt.savefig(directory_path + Title + '.png')

# Show the plot
plt.show()


# Store regression coefficients and p-values
regression_info = {}

# Loop through each unique group in 'Grupo' column
for group in data['Grupo'].unique():
    # Subset the data for that group
    subset = data[data['Grupo'] == group]

    # Fit a linear regression model and get p-value
    slope, intercept, r_value, p_value, std_err = stats.linregress(subset['Niigata'], subset['CSE'])

    # Store the slope, intercept, and p-value in the dictionary
    regression_info[group] = (slope, intercept, p_value)

# Print regression coefficients and p-values
for group, (slope, intercept, p_value) in regression_info.items():
    print(f"For {group} group:")
    print(f"y = {slope:.3f}x + {intercept:.3f}")
    print(f"p-value = {p_value:.5f}\n")

# Fit a linear regression model for the entire data
slope_all, intercept_all, r_value_all, p_value_all, std_err_all = stats.linregress(data['Niigata'], data['CSE'])

# Print regression coefficients and p-values for the entire data
print("For the entire dataset:")
print(f"y = {slope_all:.3f}x + {intercept_all:.3f}")
print(f"p-value = {p_value_all:.5f}\n")


print('Segmento de script completo - Hayo')
#%%
#%%
#--------------------------------------------------------------------------------------------------------------------
#   OUTLIER MANAGEMENT.
#--------------------------------------------------------------------------------------------------------------------
df_Small.to_excel(Output_Dir + 'Paper1_Figures/df_Small.xlsx')
print('Segmento de script completo - Hayo')


#%%
#--------------------------------------------------------------------------------------------------------------------
#   Figura 6? Matriz de Correlacion.
#--------------------------------------------------------------------------------------------------------------------
df=df_Small
df = df.rename(columns={"No Inmersivo": "CSE"})
df = df[['Niigata', 'DHI','EVA','CSE','BDI','STAI_Estado','STAI_Rasgo','MOCA','WAIS_d','WAIS_i','TMT_A_s','TMT_B_s','Corsi_d','Corsi_i','London', 'Edad','N_Educacional']]
df['MOCA'] *= -1
df['WAIS_d'] *= -1
df['WAIS_i'] *= -1
df['Corsi_d'] *= -1
df['Corsi_i'] *= -1
df['London'] *= -1
new_titles_dict = {
    'Niigata':'Niigata - PPPD symptoms',
    'DHI':'DHI - Dizziness impact on life',
    'EVA':'AVSD - Dizziness intensity',
    'CSE':'CSE - Spatial navigation error',
    'BDI':'BDI - Depressive symtpoms',
    'STAI_Estado':'STAI - State Anxiety',
    'STAI_Rasgo':'STAI - Trait Anxiety',
    'MOCA':'MoCA (-1*)- Global cognition',
    'WAIS_d':'DST (-1*)- digit span memory',
    'WAIS_i':'DST(inverted) (-1*)- digit memory',
    'TMT_A_s':'TMT A - visuospatial attention ',
    'TMT_B_s':'TMT B - visuospatial executive function',
    'Corsi_d':'CBTT (-1*)- visuospatial memory',
    'Corsi_i':'CBTT(inverted) (-1*)- visuospatial memory',
    'London':'ToL (-1*)- Visuospatial planning',
    'Edad':'Age',
    'N_Educacional':'Educational level'

}
df = df.rename(columns=new_titles_dict)

corr = np.zeros((len(df.columns), len(df.columns)))
pvals = np.zeros((len(df.columns), len(df.columns)))

for i, a in enumerate(df.columns):
    for j, b in enumerate(df.columns):
        corr[i, j], pvals[i, j] = spearmanr(df[a], df[b])
corr_matrix = pd.DataFrame(corr, index=df.columns, columns=df.columns)
pvals_matrix = pd.DataFrame(pvals, index=df.columns, columns=df.columns)

# Create a mask for non-significant p-values
mask = np.invert(np.triu(pvals_matrix<0.05))

# Plotting
plt.figure(figsize=(30, 26))
sns.heatmap(corr_matrix, annot=True, center=0, fmt=".2f", cmap="coolwarm", mask=mask, linewidths=1, linecolor="lightgray", cbar_kws={"label": "Correlation coefficient"})
Title = "Figure 6 - Spearman Correlation with Significance"
plt.title("Figure 6 - Spearman Correlations with Significance (p < 0.05)", fontsize = 32, weight='bold')
plt.xticks(rotation=45, ha="right", fontsize=26, weight='bold')  # Adjust rotation as needed; 'ha' stands for horizontal alignment
plt.yticks(fontsize=26, weight='bold')  # Adjust rotation as needed; 'ha' stands for horizontal alignment

plt.tight_layout()  # Ensure everything fits in the saved figure


directory_path = Output_Dir + 'Paper1_Figures/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
plt.savefig(directory_path + Title + '.png')
plt.show()
print('Segmento de script completo - Hayo')
#%%
print('Listoco (Script completo) - Hayo')