# ---------------------------------------------------------
# Lab ONCE - Diciembre 2022
# Fondecyt 11200469
# Hayo Breinbauer
# ---------------------------------------------------------
#%%
import pandas as pd     #Base de datos
import numpy as np
import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos
from pathlib import Path
home= str(Path.home()) # Obtener el directorio raiz en cada computador distinto
Py_Processing_Dir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Py_Processing/"



codex_df = pd.read_csv((Py_Processing_Dir+'AB_SimianMaze_Z4_Resumen_Pacientes_Analizados.csv'), index_col=0)
m_df = pd.read_csv((Py_Processing_Dir+'AB_SimianMaze_Z3_NaviDataBreve_con_calculos.csv'), index_col=0)
p_df = pd.read_csv((Py_Processing_Dir+'AB_SimianMaze_Z2_NaviData_con_posicion.csv'), index_col=0)
p_df= p_df.reset_index(drop=True)
    # Invocamos en m_df (main Dataframe) la base de datos "corta" con calculo de CSE por Trial
    # Invocamos en p_df (position Dataframe) la base con tutti cuanti - sobre todo datos posicionales




home= str(Path.home()) # Obtener el directorio raiz en cada computador distinto
BaseDir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Outputs/MorrisWMz/"
SavePlotDir = home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/Outputs/FINO/"
sns.set_palette('pastel')
pal = sns.color_palette(n_colors=3)
pal = pal.as_hex()

# Primerito una revisión rápida descriptiva de la muestra en termino de tamaño de grupos y edades
sns.set(style= 'white', palette='pastel', font_scale=2,rc={'figure.figsize':(12,12)})
Mi_Orden = ['MPPP', 'Vestibular', 'Voluntario Sano']

ax = sns.countplot(data= codex_df, x= 'Grupo', order=Mi_Orden)
ax.bar_label(ax.containers[0])
ax.set_title('Conteo de Pacientes por Grupo')
plt.savefig(BaseDir+'001_Conteo_por_Grupo.png')
plt.show()

ax = sns.boxplot(data= codex_df, x= 'Grupo', y = 'Edad', showmeans=True,
                 meanprops={"marker":"o",
                       "markerfacecolor":"white",
                       "markeredgecolor":"black",
                      "markersize":"10"}, order=Mi_Orden).set(title=('Edad de Pacientes en cada Grupo'))
plt.savefig(BaseDir+'002_Edad_por_Grupo.png')
plt.show()

#-------------------------------------------------------------------------------
#  @@@ Aqui primero una preparación adicional de la base de datos

m_df['Main_Block']=m_df['True_Block']  # Aqui vamos a recodificar los Bloques en un "Main Block" más grueso
m_df.loc[(m_df.Main_Block == 'FreeNav'),'Main_Block']='Non_relevant'
m_df.loc[(m_df.Main_Block == 'Training'),'Main_Block']='Non_relevant'
m_df.loc[(m_df.Main_Block == 'VisibleTarget_1'),'Main_Block']='Target_is_Visible'
m_df.loc[(m_df.Main_Block == 'VisibleTarget_2'),'Main_Block']='Target_is_Visible'
m_df.loc[(m_df.Main_Block == 'HiddenTarget_1'),'Main_Block']='Target_is_Hidden'
m_df.loc[(m_df.Main_Block == 'HiddenTarget_2'),'Main_Block']='Target_is_Hidden'
m_df.loc[(m_df.Main_Block == 'HiddenTarget_3'),'Main_Block']='Target_is_Hidden'

r_df = m_df.loc[ m_df['Main_Block']!='Non_relevant'] # Seleccionamos solo los relevantes
rNI_df = r_df.loc[ r_df['Modalidad']=='No Inmersivo'] # rNI_df es la base para lo No Inmersivo
rRV_df = r_df.loc[ r_df['Modalidad']!='No Inmersivo'] # rRV_df es la base para lo Realidad Virtual

p_df['Main_Block']=m_df['True_Block']  # Aqui vamos a recodificar los Bloques en un "Main Block" más grueso
p_df.loc[(p_df.Main_Block == 'FreeNav'),'Main_Block']='Non_relevant'
p_df.loc[(p_df.Main_Block == 'Training'),'Main_Block']='Non_relevant'
p_df.loc[(p_df.Main_Block == 'VisibleTarget_1'),'Main_Block']='Target_is_Visible'
p_df.loc[(p_df.Main_Block == 'VisibleTarget_2'),'Main_Block']='Target_is_Visible'
p_df.loc[(p_df.Main_Block == 'HiddenTarget_1'),'Main_Block']='Target_is_Hidden'
p_df.loc[(p_df.Main_Block == 'HiddenTarget_2'),'Main_Block']='Target_is_Hidden'
p_df.loc[(p_df.Main_Block == 'HiddenTarget_3'),'Main_Block']='Target_is_Hidden'

pos_df = p_df.loc[ p_df['Main_Block']!='Non_relevant'] # Seleccionamos solo los relevantes
posNI_df = pos_df.loc[ pos_df['Modalidad']=='No Inmersivo'] # posNI_df es la base para lo No Inmersivo
posRV_df = pos_df.loc[ pos_df['Modalidad']!='No Inmersivo'] # posNI_df es la base para lo Realidad Virtual

#%%
show_df = m_df.loc[ m_df['True_Block']=='HiddenTarget_2']

ax= sns.barplot(data=show_df, x='Grupo', y='CSE', hue='Modalidad')
plt.show()

#%%
show_df = m_df.loc[ m_df['Grupo']=='Voluntario Sano']
sns.set(style= 'white', palette='pastel',font_scale=2, rc={'figure.figsize':(21,12)})
ax=sns.barplot(data=show_df, x='True_Block', y='CSE', hue='Modalidad').set(title=('Voluntario Sano'))
plt.show()

show_df = m_df.loc[ m_df['Grupo']=='MPPP']
sns.set(style= 'white', palette='pastel',font_scale=2, rc={'figure.figsize':(21,12)})
ax=sns.barplot(data=show_df, x='True_Block', y='CSE', hue='Modalidad').set(title=('MPPP'))
plt.show()

show_df = m_df.loc[ m_df['Grupo']=='Vestibular']
sns.set(style= 'white', palette='pastel',font_scale=2, rc={'figure.figsize':(21,12)})
ax=sns.barplot(data=show_df, x='True_Block', y='CSE', hue='Modalidad').set(title=('Vestibular'))
plt.show()
#%%
BanishList=['P01','P02','P03','P04']
clear_df = m_df[~m_df['Sujeto'].isin(BanishList)]
ax= sns.barplot(data=clear_df, x='True_Block', y='True_Trial', hue='Grupo')
plt.show()

#%%
#-------------------------------------------------------------------------------

# Revisión general

sns.set(style= 'white', palette='pastel',font_scale=1, rc={'figure.figsize':(12,12)})
ax = sns.boxplot(x='Grupo',y='CSE',hue='Sujeto',data=rNI_df, order=Mi_Orden).set(title=('CSE global todos los trials'))
plt.savefig(BaseDir+'010_CSE_Global_con_Pxx.png')
plt.show()

ax = sns.boxplot(x='Sujeto',y='CSE',hue='Grupo',data=rNI_df).set(title=('CSE global todos los trials'))
plt.savefig(BaseDir+'011_CSE_Global_POR_Pxx.png')
plt.show()

sns.set(style= 'white', palette='pastel',font_scale=2, rc={'figure.figsize':(12,12)})

def Plot_CSE_BoxPlot(dat,Column,Condition):
    show_df = dat.loc[rNI_df[Column] == Condition]
    titulo='CSE por Grupo cuando '+ Column +' es ' + Condition
    ax=sns.boxplot(data=show_df, x='Grupo', y='CSE',showmeans=True,
                 meanprops={"marker":"o",
                       "markerfacecolor":"white",
                       "markeredgecolor":"black",
                      "markersize":"10"}, order=Mi_Orden)
    ax.set_title(titulo)
    plt.savefig(BaseDir + '012_'+titulo+'.png')
    plt.show()

Plot_CSE_BoxPlot(rNI_df,'Main_Block','Target_is_Visible')
show_data = rNI_df.loc[rNI_df['True_Trial']==2]
Plot_CSE_BoxPlot(show_data,'True_Block','VisibleTarget_2')
Plot_CSE_BoxPlot(rNI_df,'Main_Block','Target_is_Hidden')



def Plot_Mwz_Learning(Scope,Bloque,dat):
    plot_df=dat.loc[dat[Scope]==Bloque]
    sns.set(style='white',font_scale=2, rc={'figure.figsize': (12, 12)})
    ax = sns.lineplot(data=plot_df, x='True_Trial',y='CSE',hue='Grupo', hue_order=Mi_Orden).set(title=('Learning_at_' + Scope + '_'+ Bloque))
    plt.savefig(BaseDir + '020_Learning_at_' + Scope + '_'+ Bloque +'.png')
    plt.show()

Plot_Mwz_Learning('True_Block','VisibleTarget_1',rNI_df)
Plot_Mwz_Learning('True_Block','VisibleTarget_2',rNI_df)
Plot_Mwz_Learning('Main_Block','Target_is_Visible',rNI_df)
Plot_Mwz_Learning('True_Block','HiddenTarget_1',rNI_df)
Plot_Mwz_Learning('True_Block','HiddenTarget_2',rNI_df)
Plot_Mwz_Learning('True_Block','HiddenTarget_3',rNI_df)
Plot_Mwz_Learning('Main_Block','Target_is_Hidden',rNI_df)

sns.set(style='white', palette='pastel', font_scale=2, rc={'figure.figsize': (12, 12)})

# Mapa de Calor
#-------------------------------------------------------------------------------
#%%
# MAPAS DE CALOR DEF


def MapaDeCalor(dat,num,Titulo):
    colores = sns.color_palette('coolwarm',n_colors=100)
    dat.reset_index()
    sns.set_style("ticks")
    sns.set(style='ticks',rc={"axes.facecolor":colores[0]},font_scale=1.5)
    ax = sns.kdeplot(data=dat, x='P_position_x', y='P_position_y', cmap='coolwarm', n_levels=100, thresh=0, fill=True, cbar=True)

    sns.set_context(font_scale=3)
    ax.set(ylim=(-0.535, 0.535), xlim=(-0.535, 0.535), aspect=1)
    ax.tick_params(labelsize=13)
    ax.set_title(Titulo, fontsize=22)
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
        (show_df['platformPosition_x'].iloc[0] - (PSize / 2), show_df['platformPosition_y'].iloc[0] - (PSize / 2)),
        PSize, PSize, linewidth=2.5, edgecolor='yellow',linestyle='--',
        facecolor='none')

    ax.add_artist(rectA)
    plt.savefig(SavePlotDir + num +Titulo+'.png')

    plt.show()

BanishList=['P01','P03', 'P05', 'P24', 'P30','P35']
#BanishList=['P01','P03', 'P05', 'P07', 'P12', 'P23', 'P24', 'P25', 'P35', 'P30']

pre_df = posNI_df.loc[posNI_df['True_Block']=='HiddenTarget_1']
pre_df = pre_df[~pre_df['Sujeto'].isin(BanishList)]
show_df = pre_df.loc[pre_df['Grupo']=='MPPP']
MapaDeCalor(show_df,'1FINO-NI', 'MPPP')
show_df = pre_df.loc[pre_df['Grupo']=='Vestibular']
MapaDeCalor(show_df,'1FINO-NI', 'Vestibular (no-MPPP)')
show_df = pre_df.loc[pre_df['Grupo']=='Voluntario Sano']
MapaDeCalor(show_df,'1FINO-NI', 'Voluntario Sano')


pre_df = posRV_df.loc[posRV_df['True_Block']=='HiddenTarget_1']
pre_df = pre_df[~pre_df['Sujeto'].isin(BanishList)]
show_df = pre_df.loc[pre_df['Grupo']=='MPPP']
MapaDeCalor(show_df,'1FINO-RV', 'MPPP')
show_df = pre_df.loc[pre_df['Grupo']=='Vestibular']
MapaDeCalor(show_df,'1FINO-RV', 'Vestibular (no-MPPP)')
show_df = pre_df.loc[pre_df['Grupo']=='Voluntario Sano']
MapaDeCalor(show_df,'1FINO-RV', 'Voluntario Sano')


#%%
show_df= r_df[~r_df['Sujeto'].isin(BanishList)]
show_df = show_df.loc[show_df['True_Block']=='HiddenTarget_1']
sns.set(style= 'white', palette='pastel', font_scale=2.5,rc={'figure.figsize':(18,12)})

ax=sns.boxplot(data=show_df, x='Modalidad', y='CSE',hue='Grupo',hue_order=Mi_Orden).set(title=('Error Acumulado en Navegación'))
#sns.set(ylabel='Error acumulado de búsqueda/Navegación')
#ax.set(xlabel='Modalidad de la prueba de Navegación')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.ylim(0,300)
plt.xlabel('Modalidad')
plt.ylabel('Error acumulado de búsqueda')
plt.savefig(SavePlotDir + '1FINO-resumen.png', bbox_inches='tight')
plt.show()

#%%
# Tesis Rosario
# Hagamos un conteo de los Trials logrados en RV.
sns.set(style= 'white', palette='pastel',font_scale=1, rc={'figure.figsize':(12,12)})

Titulo = 'Trials completados en Realidad virtual (conteo x Grupo)'
ax = sns.countplot(data = rRV_df, x= 'Grupo', hue='Sujeto').set(title=(Titulo))
plt.savefig(BaseDir + 'Tesis_Rosario_'+Titulo+'.png')
plt.show()

Titulo = 'Trials completados en Realidad virtual (conteo x Sujeto)'
ax = sns.countplot(data = rRV_df, x= 'Sujeto', hue='Grupo').set(title=(Titulo))
plt.savefig(BaseDir + 'Tesis_Rosario_'+Titulo+'.png')
plt.show()
#%%
# Este pedazo de codigo es para poder jugar con los Trials logrados

u = rRV_df.loc[rRV_df['Sujeto']!='P01']   # Esto es como P01 tiene menos trials por haber sido el piloto, lo eliminamos del analisis

#Más elegante para poder eliminar Sujetos del Analisis que queramos limpiar:

BanishList=['P01']
clear_df = r_df[~r_df['Sujeto'].isin(BanishList)] # Aqui tendríamos una base de datos sin los
                    # sujetos de la Banish List... ojo aún no ocupo clear_df

Trial_Count = u.groupby('Grupo')['Sujeto'].value_counts().to_frame()
Trial_Count.index = Trial_Count.index.set_names(['Grupo', 'Suj'])
Trial_Count = Trial_Count.reset_index(level=[1])
Trial_Count.reset_index(inplace=True)
Trial_Count= Trial_Count.rename(columns = {'Sujeto':'Trials_logrados'})
Titulo ='Resumen De Trials Logrados por Grupo'
ax = sns.boxplot(data= Trial_Count, x= 'Grupo', y = 'Trials_logrados', showmeans=True,
                 meanprops={"marker":"o",
                       "markerfacecolor":"white",
                       "markeredgecolor":"black",
                      "markersize":"10"}, order=Mi_Orden).set(title=(Titulo))
plt.savefig(BaseDir + 'Tesis_Rosario_'+Titulo+'.png')
plt.show()

#%%

# Ahora veamos, y primero sin filtrar por Trials Logrados, como se comportan las variables principales
Titulo = 'CSE por Grupo y Modalidad (RV vs NI)'
ax = sns.boxplot(data= r_df, x= 'Grupo', y = 'CSE', hue='Modalidad', showmeans=True,
                 meanprops={"marker":"o",
                       "markerfacecolor":"white",
                       "markeredgecolor":"black",
                      "markersize":"10"}, order=Mi_Orden).set(title=(Titulo))
plt.ylim(0,350)
plt.savefig(BaseDir + 'Tesis_Rosario_'+Titulo+'.png')
plt.show()

Titulo = 'CSE por Grupo y Modalidad (RV vs NI) con Target Hidden'
df = r_df.loc[r_df['True_Block']=='HiddenTarget_3']
ax = sns.boxplot(data= df, x= 'Grupo', y = 'CSE', hue='Modalidad', showmeans=True,
                 meanprops={"marker":"o",
                       "markerfacecolor":"white",
                       "markeredgecolor":"black",
                      "markersize":"10"}, order=Mi_Orden).set(title=(Titulo))
plt.ylim(0,350) # Esto es para hacer el gráfico más legible. Es para poner el corte de CSE en 350 para
                # la lectura del gráfico y complicar la lectura con CSE muy altos de un par
                # de Outliers
plt.savefig(BaseDir + 'Tesis_Rosario_'+Titulo+'.png')
plt.show()

def Plot_Mwz_Learning_L(Modalidad,Scope,Bloque,dat, Titulo):
    plot_df=dat.loc[dat[Scope]==Bloque]
    plot_df=plot_df.loc[dat['Modalidad']==Modalidad]
    sns.set(style='white',font_scale=2, rc={'figure.figsize': (12, 12)})
    ax = sns.lineplot(data=plot_df, x='True_Trial',y='CSE',hue='Grupo', hue_order=Mi_Orden).set(title=('Learning_at_' + Scope + '_'+ Bloque))
    plt.savefig(BaseDir + 'Tesis_Rosario_'+ Titulo + '.png')
    plt.show()

Plot_Mwz_Learning_L('No Inmersivo','True_Block','HiddenTarget_3', r_df, 'Aprendizaje en No inmersivo')
Plot_Mwz_Learning_L('Realidad Virtual','True_Block','HiddenTarget_3', r_df, 'Aprendizaje en No inmersivo')

#%%
#-------------------------------------------------------------------------------
# Realizaremos estudio de caso a caso.

PXX_List = rNI_df['Sujeto'].unique()  # Obtenemos una lista de todos los sujetos individualizados

for Pxx in PXX_List:     # Vamos ahora sujeto por sujeto.
    print(Pxx)
    show_df = rNI_df.loc[rNI_df['Sujeto']==Pxx]
    showPos_df = posNI_df.loc[posNI_df['Sujeto']==Pxx]
    ax = sns.countplot(x='True_Block',hue='True_Trial', data= show_df).set(title=('Conteo de Trials en ' + str(Pxx)))
    plt.show()
    print('Next')

#-------------------------------------------------------------------------------
# Ahora intentaremos replicar el estudio original

# Primero nos aseguramos que tengamos todos los trial en este grupo
Trial_Count = rNI_df['Sujeto'].value_counts(ascending = True)
# Y de inmediato tenemos un problema... no todos los sujetos tienen el mismo numero de Trials!
# PORQUE???
# Primero mejor realizaremos el estudio detallado por cada individuo por cada trial



show_df = m_df.loc[ m_df['Main_Block']!='Non_relevant']
g = sns.FacetGrid(show_df, row="Main_Block", col="Modalidad")
g.figure.set_size_inches(17,17)
g.map(sns.lineplot, "True_Trial", "CSE", "Grupo")
g.add_legend()
plt.show()

show_df = m_df.loc[ m_df['Main_Block']=='Target_is_Hidden']
g = sns.FacetGrid(show_df, row="True_Block", col="Modalidad")
g.figure.set_size_inches(17,17)
g.map(sns.lineplot, "True_Trial", "CSE", "Grupo")
g.add_legend()
plt.show()

#show_df = m_df.loc[ m_df['True_Block']=='VisibleTarget_1']
#show_df = m_df.loc[ m_df['Modalidad']=='No Inmersivo']

ax = sns.barplot(x='Sujeto',y='CSE',hue='Grupo', data=m_df)
ax.figure.set_size_inches(17,17)
plt.show()

g = sns.FacetGrid(m_df, row="True_Block", col="Modalidad")
g.figure.set_size_inches(17,17)

g.map(sns.lineplot, "True_Trial", "CSE", "Grupo")
g.add_legend()
plt.show()

show_df = m_df.loc[ m_df['Modalidad']=='No Inmersivo']
ax = sns.barplot(x="Grupo",y="CSE",data=show_df)
plt.show()

show_df = m_df.loc[ m_df['Modalidad']=='No Inmersivo']
show_df = show_df.loc[ m_df['Grupo']=='MPPP']
ax = sns.barplot(x="Sujeto",y="CSE",data=show_df)
plt.show()

#g=sns.FacetGrid(data=m_df,col='True_Block', hue='Modalidad')
#g.map(sns.barplot, 'Grupo', 'CSE', order=['Voluntario Sano', 'Vestibular', 'MPPP'])
#plt.show()

show_df = m_df.loc[m_df['True_Block']=='VisibleTarget_1']
ax = sns.barplot(x="Grupo",y="CSE", hue='Modalidad', order=['Voluntario Sano', 'Vestibular', 'MPPP'],data=show_df).set(title='Target Visible (Control)')
plt.show()
ax= sns.lineplot(x="True_Trial", y="CSE", hue='Grupo', data= show_df)
plt.show()

show_df = m_df.loc[m_df['True_Block']=='VisibleTarget_2']
ax = sns.barplot(x="Grupo",y="CSE", hue='Modalidad', order=['Voluntario Sano', 'Vestibular', 'MPPP'],data=show_df).set(title='Target Visible (Control) 2')
plt.show()
ax= sns.lineplot(x="True_Trial", y="CSE", hue='Grupo', data= show_df)
plt.show()

show_df = m_df.loc[m_df['True_Block']=='VisibleTarget_2']
show_df = show_df.loc[show_df['Grupo']=='MPPP']
ax = sns.barplot(x="Sujeto",y="CSE", hue='Modalidad',data=show_df).set(title='Target Visible (Control) 2')
plt.show()
ax= sns.lineplot(x="True_Trial", y="CSE", hue='Grupo', data= show_df)
plt.show()

show_df = m_df.loc[m_df['True_Block']=='HiddenTarget_1']
ax = sns.barplot(x="Grupo",y="CSE", hue='Modalidad', order=['Voluntario Sano', 'Vestibular', 'MPPP'],data=show_df).set(title='Plataforma escondida (Desafío de navegación)')
plt.show()
ax= sns.lineplot(x="True_Trial", y="CSE", hue='Grupo', data= show_df)
plt.show()

show_df = m_df.loc[m_df['True_Block']=='HiddenTarget_2']
ax = sns.barplot(x="Grupo",y="CSE", hue='Modalidad', order=['Voluntario Sano', 'Vestibular', 'MPPP'],data=show_df).set(title='Plataforma escondida (Desafío de navegación)')
plt.show()
ax= sns.lineplot(x="True_Trial", y="CSE", hue='Grupo', data= show_df)
plt.show()

show_df = m_df.loc[m_df['True_Block']=='HiddenTarget_3']
ax = sns.barplot(x="Grupo",y="CSE", hue='Modalidad', order=['Voluntario Sano', 'Vestibular', 'MPPP'],data=show_df).set(title='Plataforma escondida (Desafío de navegación)')
plt.show()
ax= sns.lineplot(x="True_Trial", y="CSE", hue='Grupo', data= show_df)
plt.show()

#ax = sns.barplot(x="True_Block",y="CSE", hue='Grupo',data=show_df)
#ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
#plt.show()


#ax = sns.barplot(x="Grupo",y="CSE", hue='Sujeto',data=show_df)
#ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
#plt.show()
#------------------------------
#   Micro script para obtener los valores máximos de la vuelta olímpica
#   Necesita una recorrida limpia de toda la piscina en circular
#t_df = m_df['P_position_x']
#max_x = t_df.max()
#print('Maximo: ',max_x)
#   Como resultado en nuestra versión de Simian obtuvimos que la piscina mide
#   280 unidades originales de radio.
#------------------------------


#GRAFICO PATH"
Subj='P06'
Mod ='No_Inmersivo'
Bloc='HiddenTarget_3'
Plat=True
show_df = m_df.loc[ (m_df['Sujeto']==Subj) & (m_df['Modalidad']==Mod) & (m_df['True_Block']==Bloc)]
ax = sns.lineplot(x="P_position_x", y="P_position_y",hue="True_Trial", data=show_df, linewidth=3, alpha = 0.8, legend='full', palette = sns.color_palette('Blues', as_cmap = True), sort= False) #
#sns.set_palette('Blues',6)

sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":18})
sns.set_style('whitegrid')
circle= plt.Circle((0,0),0.5, color='b',fill=False)
ax.add_artist(circle)
ax.set(ylim=(-0.535,0.535),xlim=(-0.535,0.535), aspect=1) #incluir el equivalente al cuadrado de la pieza completa.
#el 0.535 considera los 300 pixeles que creo que mide la caja de Simian desde punto centro, divido en los 560 pixeles de Pool Diamater
ax.set(xlabel = 'East-West (virtual units in Pool-Diameters)', ylabel= 'North-South (virtual units in Pool-Diameters)', title = 'Test-Run')
plt.xlabel('East-West (virtual units in Pool-Diameters)',fontsize=18)
plt.ylabel('North-South (virtual units in Pool-Diameters)',fontsize=18)
ax.figure.set_size_inches(7,7)
plt.xticks(np.arange(-0.5, 0.75, 0.25))
plt.yticks(np.arange(-0.5, 0.75, 0.25))
ax.tick_params(labelsize = 13)
ax.legend(frameon = False, loc='right', bbox_to_anchor=(1.3,0.5), fontsize = 13)
if Plat:
    PSize = (100 / 560)
    rectA = plt.Rectangle((show_df['platformPosition.x'].iloc[0] - (PSize / 2), show_df['platformPosition.y'].iloc[0] - (PSize / 2)), PSize, PSize, linewidth=1, edgecolor='b',
                          facecolor='none')

    ax.add_artist(rectA)

plt.show()

Subj='P06'
Mod ='No Inmersivo'
Bloc='HiddenTarget_3'
Blocs=['FreeNav','Training','VisibleTarget_1','VisibleTarget_2','HiddenTarget_1','HiddenTarget_2','HiddenTarget_3']
count = 0
for B in Blocs:

    if count > 1:
        Plat=True
    else:
        Plat=False

    show_df = m_df.loc[ (m_df['Sujeto']==Subj) & (m_df['Modalidad']==Mod) & (m_df['True Block']==B)]
    ax=0
    ax = sns.lineplot(x="P_position_x", y="P_position_y",hue="True Trial", data=show_df, linewidth=3, alpha = 0.8, legend='full', sort= False)
    if count > 1:
        ax = sns.lineplot(x="P_position_x", y="P_position_y", hue="True Trial", data=show_df, linewidth=3, alpha=0.8, legend='full',palette = sns.color_palette('Blues', as_cmap = True), sort=False)
    sns.set_context("paper", rc={"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 18})
    sns.set_style('whitegrid')
    circle = plt.Circle((0, 0), 0.5, color='b', fill=False)
    ax.add_artist(circle)
    ax.set(ylim=(-0.535, 0.535), xlim=(-0.535, 0.535), aspect=1)  # incluir el equivalente al cuadrado de la pieza completa.
    # el 0.535 considera los 300 pixeles que creo que mide la caja de Simian desde punto centro, divido en los 560 pixeles de Pool Diamater
    ax.set(xlabel='East-West (virtual units in Pool-Diameters)', ylabel='North-South (virtual units in Pool-Diameters)', title=(Subj + '-' + Mod +'-'+ B))
    plt.xlabel('East-West (virtual units in Pool-Diameters)', fontsize=18)
    plt.ylabel('North-South (virtual units in Pool-Diameters)', fontsize=18)
    ax.figure.set_size_inches(7, 7)
    plt.xticks(np.arange(-0.5, 0.75, 0.25))
    plt.yticks(np.arange(-0.5, 0.75, 0.25))
    ax.tick_params(labelsize=13)
    ax.legend(frameon=False, loc='right', bbox_to_anchor=(1.3, 0.5), fontsize=13)
    if Plat:
        PSize= (100 / 560)
        rectA = plt.Rectangle((show_df['platformPosition.x'].iloc[0]-(PSize/2), show_df['platformPosition.y'].iloc[0]-(PSize/2)), PSize, PSize, linewidth=1, edgecolor='b',
                              facecolor='none')

        ax.add_artist(rectA)
    plt.show()
    count+=1

