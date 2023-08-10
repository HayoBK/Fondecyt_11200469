# ---------------------------------------------------------
# Lab ONCE - Agosto 2023
# Fondecyt 11200469
# Hayo Breinbauer
# ---------------------------------------------------------
#%%
import pandas as pd     #Base de datos
import numpy as np
import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos
from pathlib import Path
import os

#--------------------------------------------------------------------------------------------------------------------
#   PREPARAR DIRECTORIOS PARA MANEJO DE DATOS
#--------------------------------------------------------------------------------------------------------------------


home= str(Path.home()) # Obtener el directorio raiz en cada computador distinto
Py_Processing_Dir= home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"
Output_Dir=        home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Outputs/FENRIR/"

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

Mod_List=['No Inmersivo','Realidad Virtual']
Block_List = ['FreeNav','Training','VisibleTarget_1','VisibleTarget_2','HiddenTarget_1','HiddenTarget_2','HiddenTarget_3']

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
                PSize, PSize, linewidth=1, edgecolor='b',
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


#%%
#--------------------------------------------------------------------------------------------------------------------
#   Correr los análisis propiamente tal.
#--------------------------------------------------------------------------------------------------------------------

PathGraph(df_Pos, 'P06', 'No Inmersivo', 'HiddenTarget_1', 'Test')

for index, value in df_Small['Sujeto'].iteritems():
    i=0
    for M in Mod_List:
        for B in Block_List:
            i+=1
            PathGraph(df_Pos, value, M, B, ('Fig_'+str(i)))

#--------------------------------------------------------------------------------------------------------------------
#   End of File
#--------------------------------------------------------------------------------------------------------------------

print('Listoco - Hayo')