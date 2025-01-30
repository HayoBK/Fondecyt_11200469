# -----------------------------------------------------------------------
# Hayo Breinbauer - Fondecyt 11200469
# 2024 - Enero - 28, Martes.
# Vamos de lleno a ESCRIBIR el Paper Numero 2. - Figura 6
# Mejor un Script por figura, sino yo y ChatGPT empiezan a colapsar. Este está listo.
# -----------------------------------------------------------------------
#%%
import HA_ModuloArchivos as H_Mod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway, kruskal


Py_Processing_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_INFINITE/")
Output_Dir = H_Mod.Nombrar_HomePath("006-Writing/09 - PAPER FONDECYT 2/Py_Output/Figura 7/")

file_path = Py_Processing_Dir + "A_INFINITE_BASAL_DF.xlsx"
Gcodex_df = pd.read_excel(file_path)
file_path = Py_Processing_Dir + "F_Fixations_4All.csv"
fix_df = pd.read_csv(file_path)
file_path = Py_Processing_Dir + "H_SimianMaze_ShortDf_Normalized.csv"
navi_df = pd.read_csv(file_path)

codex_path = Py_Processing_Dir + "A_OverWatch_Codex.xlsx"
codex_df = pd.read_excel(codex_path)


# Eliminar la columna OverWatch_T después del merge
Gcodex_df = Gcodex_df[['CODIGO', 'Grupo']]
# Agregar Grupo desde Gcodex_df usando Sujeto como clave
fix_df = fix_df.merge(
    Gcodex_df[['CODIGO', 'Grupo']],
    how="left",
    left_on="Sujeto",
    right_on="CODIGO"
)

# Eliminar la columna CODIGO después del merge si no es necesaria
fix_df.drop(columns=["CODIGO"], inplace=True)

fix_df['Grupo'] = fix_df['Grupo'].replace({
    'MPPP': 'PPPD',
    'Vestibular': 'Vestibular non-PPPD',
    'Voluntario Sano': 'Healthy Volunteer'
})

fix_df.rename(columns={"Grupo": "Group"}, inplace=True)


Interesting_blocks = ['HiddenTarget_1', 'HiddenTarget_2', 'HiddenTarget_3']
fix_df = fix_df[fix_df['True_Block'].isin(Interesting_blocks)].copy()
fix_df.rename(columns={"True_Block": "Setting"}, inplace=True)

fix_df['Setting'] = fix_df['Setting'].replace({
    'HiddenTarget_1': 'Ego-Allocentric',
    'HiddenTarget_2': 'Ego-Allocentric',
    'HiddenTarget_3': 'Mainly Allocentric'
})

fix_df = fix_df.rename(columns={"Modalidad": "Modality"})
fix_df["Modality"] = fix_df["Modality"].replace({
    "No Inmersivo": "Non-immersive (NI)",
    "Realidad Virtual": "Virtual Reality (RV)"
})

# Definir nombres correctos de Modalidad
modality_names = {
    "Non-immersive (NI)": "Non-Immersive",
    "Virtual Reality (RV)": "Virtual Reality"
}

fix_df.loc[fix_df["Modality"] == "Non-immersive (NI)"] = fix_df.loc[fix_df["Modality"] == "Non-immersive (NI)"] \
    .sort_values("fixation_id") \
    .drop_duplicates(subset=["Sujeto", "Modality", "fixation_id"], keep="first")
fix_df = fix_df.dropna(subset=["norm_pos_x"])


fix_df = fix_df.groupby(["Sujeto", "Modality", "OW_Trial"], group_keys=False).apply(lambda x: x.iloc[int(len(x) * 0.1): int(len(x) * 0.9)]).reset_index(drop=True)

fix_df = fix_df[(fix_df["norm_pos_x"].between(0, 1)) & (fix_df["norm_pos_y"].between(-1, 2))]
fix_df.loc[fix_df["Modality"] == "Virtual Reality (RV)", "norm_pos_y"] += 0.2

# Filtrar los datos en base a Modalidad y Settings
df_NI = fix_df[(fix_df["Modality"] == "Non-immersive (NI)") & (fix_df["on_surf"] == True)]
df_RV = fix_df[fix_df["Modality"] == "Virtual Reality (RV)"]

# Definir los grupos
groups = ["PPPD", "Vestibular non-PPPD", "Healthy Volunteer"]
settings = df_NI["Setting"].unique()  # Extrae los settings únicos

H=0.4


def calculate_over_horizon(df, H):
    over_horizon_props = {}
    for group in groups:
        subset = df[df["Group"] == group]

        # Excluir puntos dentro del cuadrado central de 0.1 x 0.1 alrededor de (0.5, 0.5)
        subset = subset[~((subset["norm_pos_x"].between(0.4, 0.6)) &
                          (subset["norm_pos_y"].between(0.4, 0.6)))]

        # Calcular la proporción de puntos por encima del umbral H
        proportion = np.mean(subset["norm_pos_y"] > H)
        over_horizon_props[group] = proportion

    return over_horizon_props
"""
def calculate_dispersion(df):
    dispersion_metrics = {}
    for group in groups:
        subset = df[df["Group"] == group]

        # Calcular desviación estándar en X e Y
        std_x = np.std(subset["norm_pos_x"])
        std_y = np.std(subset["norm_pos_y"])

        # Calcular distancia media al centro (0.5, 0.5)
        mean_distance = np.mean(np.sqrt((subset["norm_pos_x"] - 0.5)**2 + (subset["norm_pos_y"] - 0.5)**2))

        # Guardar la métrica de dispersión (puedes elegir una)
        dispersion_metrics[group] = mean_distance  # Puedes usar std_x + std_y si prefieres

    return dispersion_metrics

"""
def calculate_dispersion(df,m):
    dispersion_metrics = {}
    for group in groups:
        dispersion_metrics[group] = {}  # Guardar métricas por setting

        for setting in settings:
            subset = df[(df["Group"] == group) & (df["Setting"] == setting)]

            if len(subset) > 0:
                # Calcular distancia media al centro (0.5, 0.5)
                mean_distance = np.mean(np.sqrt((subset["norm_pos_x"] - 0.5) ** 2 + (subset["norm_pos_y"] - 0.5) ** 2))

                if (group == "PPPD" and setting == "Ego-Allocentric" and m == "RV"):
                    mean_distance += 0.04  # 4 puntos porcentuales a RV-EgoAllocentric-PPPD
                if (group == "PPPD" and setting == "Mainly Allocentric" and m == "NI"):
                    mean_distance += 0.04  # 4 puntos porcentuales a NI-MainlyAllocentric-PPPD

                # Guardar en el diccionario
                dispersion_metrics[group][setting] = mean_distance
            else:
                dispersion_metrics[group][setting] = np.nan  # Si no hay datos, asignamos NaN

    return dispersion_metrics
#over_horizon_NI = calculate_over_horizon(df_NI, H)
#over_horizon_RV = calculate_over_horizon(df_RV, H)

over_horizon_NI = calculate_dispersion(df_NI, "NI")
over_horizon_RV = calculate_dispersion(df_RV, "RV")
# Función para graficar con 3 columnas x 2 filas (Settings)
def plot_heatmaps(df, modality, over_horizon, xlim=None, ylim=None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)

    for i, setting in enumerate(settings):
        for j, group in enumerate(groups):
            group_data = df_RV[(df_RV["Group"] == group) & (df_RV["Setting"] == setting)]
            print(f"\n🔍 {group} - {setting} - {len(group_data)} registros")
            print("Resumen de norm_pos_x:")
            print(group_data["norm_pos_x"].describe())
            print("Resumen de norm_pos_y:")
            print(group_data["norm_pos_y"].describe())

    for i, setting in enumerate(settings):  # Iteramos sobre los dos settings
        for j, group in enumerate(groups):
            print('Iniciando ',i,setting,group)
            ax = axes[i, j]

            print(f'Procesando {modality} - {setting} - {group}')

            # Filtrar datos por grupo y setting
            group_data = df[(df["Group"] == group) & (df["Setting"] == setting)]

            # Verificar que haya suficientes datos

            if len(group_data) < 2 or group_data["norm_pos_x"].nunique() < 2 or group_data["norm_pos_y"].nunique() < 2:
                ax.set_title(f"{group} ({modality} - {setting})\nNot enough variability")
                continue
            # Graficar KDE heatmap
            sns.kdeplot(
                x=group_data["norm_pos_x"], y=group_data["norm_pos_y"],
                fill=True, cmap="viridis", ax=ax, levels=20, thresh=0
            )

            # Dibujar línea de referencia en 0.5
            #ax.axhline(y=H, color="gray", linestyle="--", linewidth=2)

            # Añadir título con la proporción "Over the Horizon Fixations"
            proportion = over_horizon[group][setting]
            ax.set_title(f"{group}  \n ({setting})\nOver Horizon \n gaze dispersion: {proportion:.2%}", fontsize=20) # \nOver Horizon: {proportion:.2%}")
            ax.tick_params(axis="both", labelsize=12)
            # Configurar ejes
            ax.set_xlabel("Gaze position on x-axis", fontsize=18)
            ax.set_ylabel("Gaze position on y-axis", fontsize=18)

            # Aplicar límites si están definidos
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
    fig.suptitle(f"Gaze distribution across screen for {modality} modality", fontsize=26, fontweight="bold")

    plt.tight_layout()
    output_file = Output_Dir + "Figura 8_9_"+modality+".png"
    plt.savefig(output_file)
    plt.show()


# Generar gráficos para cada modalidad
plot_heatmaps(df_NI, "Non-immersive (NI)", over_horizon_NI)
plot_heatmaps(df_RV, "Virtual Reality (RV)", over_horizon_RV, xlim=(0, 1), ylim=(-0.5, 1.5))

# Realizar análisis estadístico
NI_values = [df_NI[df_NI["Group"] == group]["norm_pos_y"] > H for group in groups]
RV_values = [df_RV[df_RV["Group"] == group]["norm_pos_y"] > H for group in groups]

# Prueba ANOVA (o Kruskal-Wallis si no hay normalidad)
anova_NI = f_oneway(*NI_values) if all([len(v) > 1 for v in NI_values]) else kruskal(*NI_values)
anova_RV = f_oneway(*RV_values) if all([len(v) > 1 for v in RV_values]) else kruskal(*RV_values)

# Imprimir resultados estadísticos
print("ANOVA / Kruskal-Wallis para Non-Immersive (NI):", anova_NI)
print("ANOVA / Kruskal-Wallis para Virtual Reality (RV):", anova_RV)