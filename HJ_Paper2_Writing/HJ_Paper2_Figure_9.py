#%%
import HA_ModuloArchivos as H_Mod
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway, kruskal
import matplotlib.patches as mpatches
import scipy.stats as stats

Py_Processing_Dir = H_Mod.Nombrar_HomePath("002-LUCIEN/Py_Legacy/")
Output_Dir = H_Mod.Nombrar_HomePath("006-Writing/09 - PAPER FONDECYT 2/Py_Output/Figura 9/")


file_path = Py_Processing_Dir + "DA_Gaze_2D_reducido_All_HT.csv"
GazeNI_df = pd.read_csv(file_path)

file_path = Py_Processing_Dir + "DA_Gaze_2D_reducido_All_HT.csv"
GazeRV_df = pd.read_csv(file_path)


# Agregar columna diferenciadora a cada DataFrame
GazeNI_df["Modality"] = "Non-immersive (NI)"
GazeRV_df["Modality"] = "Virtual Reality (RV)"

# Fusionar ambos DataFrames
Gaze_df = pd.concat([GazeNI_df, GazeRV_df], ignore_index=True)

# Eliminar entradas con "Vestibular migraine"
Gaze_df = Gaze_df[Gaze_df["Categoria"] != "Vestibular Migraine"]

# Reemplazar "Vestibular (non PPPD)" por "Vestibular non PPPD"
Gaze_df["Categoria"] = Gaze_df["Categoria"].replace("Vestibular (non PPPD)", "Vestibular non PPPD")

# Renombrar la columna "Categoria" a "Group"
Gaze_df = Gaze_df.rename(columns={"Categoria": "Group"})


# Definir los bloques de interés sin 'HiddenTarget_1'
Interesting_blocks = ['HiddenTarget_1', 'HiddenTarget_3']
Gaze_df.rename(columns={"MWM_Block": "True_Block"}, inplace=True)

# Filtrar los datos manteniendo solo los bloques de interés
Gaze_df = Gaze_df[Gaze_df['True_Block'].isin(Interesting_blocks)].copy()

# Renombrar columna 'True_Block' a 'Setting'
Gaze_df.rename(columns={"True_Block": "Setting"}, inplace=True)

# Reemplazar los valores de 'Setting'
Gaze_df['Setting'] = Gaze_df['Setting'].replace({
    'HiddenTarget_1': 'Ego-Allocentric',
    'HiddenTarget_3': 'Mainly Allocentric'
})

Gaze_df = Gaze_df[~((Gaze_df["Group"] == "Vestibular non PPPD") & (Gaze_df["Scanned_Path_per_time_per_Block"] > 13))]


custom_palette = [
    "#6495ED", "#4169E1",  # Cornflower Blue, Royal Blue
    "#FF4500", "#C83700",  # Orange Red, Darker Orange Red
    "#3CB371", "#2E8B57"  # Medium Sea Green, Darker Sea Green
]

color_mapping = {
    ("PPPD", "Ego-Allocentric"): 0,
    ("PPPD", "Mainly Allocentric"): 2,
    ("Vestibular non PPPD", "Ego-Allocentric"): 4,
    ("Vestibular non PPPD", "Mainly Allocentric"): 1,
    ("Healthy Volunteer", "Ego-Allocentric"): 3,
    ("Healthy Volunteer", "Mainly Allocentric"): 5
}

color_mappingL = {
    ("PPPD", "Ego-Allocentric"): 0,
    ("PPPD", "Mainly Allocentric"): 1,
    ("Vestibular non PPPD", "Ego-Allocentric"): 2,
    ("Vestibular non PPPD", "Mainly Allocentric"): 3,
    ("Healthy Volunteer", "Ego-Allocentric"): 4,
    ("Healthy Volunteer", "Mainly Allocentric"): 5
}

hatch_mapping = {
    "Ego-Allocentric": "",
    "Mainly Allocentric": "//"
}

# Configuración de la figura en grilla de 2 filas (arriba NI, abajo RV)
fig, axes = plt.subplots(nrows=2, figsize=(14, 12), sharex=True)

modalities = ["Non-immersive (NI)", "Virtual Reality (RV)"]

for i, modality in enumerate(modalities):
    ax = axes[i]

    # Filtrar datos por modalidad
    subset_df = Gaze_df[Gaze_df["Modality"] == modality]

    # Crear boxplot
    sns.boxplot(
        data=subset_df,
        x="Group",
        y="Scanned_Path_per_time_per_Block",
        order=["PPPD", "Vestibular non PPPD", "Healthy Volunteer"],
        hue="Setting",
        hue_order=["Ego-Allocentric", "Mainly Allocentric"],
        showfliers=True,
        width=0.6,
        dodge=True,
        boxprops={'edgecolor': 'black'},
        medianprops={'color': 'black'},
        linewidth=4,
        legend= False,
        fliersize=0,
        ax=ax
    )
    ax.tick_params(axis='x', labelsize=18, rotation=0)
    ax.set_ylim(0, 80)  # Ajustar rango Y según los datos
    ax.set_ylabel("Gaze scanned path", fontsize=16)
    ax.set_xlabel("Group", fontsize=16)
    ax.set_title(f"{modality}", fontsize=20)

    # Aplicar colores y tramas manualmente
    patches = [patch for patch in ax.patches if isinstance(patch, mpatches.PathPatch)]
    for j, patch in enumerate(patches):
        group_idx = j // 2  # Dos settings por grupo
        setting_idx = j % 2

        if group_idx < 3 and setting_idx < 2:
            group = ["PPPD", "Vestibular non PPPD", "Healthy Volunteer"][group_idx]
            setting = ["Ego-Allocentric", "Mainly Allocentric"][setting_idx]

            patch.set_facecolor(custom_palette[color_mapping[(group, setting)]])
            patch.set_hatch(hatch_mapping[setting])

    # Realizar prueba de Kruskal-Wallis entre grupos dentro de cada Setting
    results = []
    for setting in ["Ego-Allocentric", "Mainly Allocentric"]:
        setting_df = subset_df[subset_df["Setting"] == setting]
        groups = [group["Scanned_Path_per_time_per_Block"].dropna().values for _, group in setting_df.groupby("Group")]

        if len(groups) > 1:
            stat, p = stats.kruskal(*groups)
            results.append((setting, stat, p))

    # Imprimir resultados estadísticos
    for setting, stat, p in results:
        print(f"Kruskal-Wallis para {modality} - {setting}: H={stat:.3f}, p={p:.4f}")

    # Añadir barras y anotaciones para p-values
    for j, group in enumerate(["PPPD", "Vestibular non PPPD", "Healthy Volunteer"]):
        subset = subset_df[subset_df["Group"] == group]
        values = [subset[subset["Setting"] == setting]["Scanned_Path_per_time_per_Block"].dropna() for setting in
                  ["Ego-Allocentric", "Mainly Allocentric"]]

        if len(values) == 2 and all(len(v) > 0 for v in values):
            stat, p = stats.kruskal(*values)
            if p < 0.05:
                x1, x2 = j - 0.2, j + 0.2
                y, h, col = 0.85, 0.02, 'black'
                ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=2, c=col)
                ax.text((x1 + x2) / 2, y + h, f"* (p={p:.3f})", ha='center', va='bottom', color=col, fontsize=12)

# Crear leyenda personalizada
legend_patches = [
    mpatches.Patch(facecolor=custom_palette[color_mappingL[(group, setting)]],
                   hatch=hatch_mapping[setting],
                   linewidth=4, edgecolor='black',
                   label=f"{group} ({setting})")
    for group in ["PPPD", "Vestibular non PPPD", "Healthy Volunteer"]
    for setting in ["Ego-Allocentric", "Mainly Allocentric"]
]
fig.legend(
    handles=legend_patches, title="Group and Setting", loc='center right', ncol=1, fontsize=14, title_fontsize=16,
    handleheight=3.2, handlelength=3.2,
    frameon=False
)

# Ajustar diseño y guardar
plt.tight_layout(rect=[0, 0, 0.65, 1])
output_file = Output_Dir + "Figura 9.png"
plt.savefig(output_file)
plt.show()

# Definir el orden de los grupos y la paleta de colores
group_order = ["PPPD", "Vestibular non PPPD", "Healthy Volunteer"]
palette = ["#6495ED", "#FF4500", "#3CB371"]  # Azul, Naranja, Verde

# Crear la figura
fig, ax = plt.subplots(figsize=(8, 6))

# Gráfico de dispersión
sns.scatterplot(
    data=Gaze_df,
    x="CSE",
    y="Scanned_Path_per_time_per_Block",
    hue="Group",
    hue_order=group_order,
    palette=palette,
    alpha=0.7,
    ax=ax
)

# Agregar líneas de regresión separadas por grupo y calcular correlaciones
correlation_results = {}
for group in group_order:
    subset = Gaze_df[Gaze_df["Group"] == group]

    if len(subset) > 1:  # Asegurar que hay datos suficientes para correlación
        r, p = stats.spearmanr(subset["CSE"], subset["Scanned_Path_per_time_per_Block"], nan_policy='omit')
        correlation_results[group] = (r, p)

        # Agregar línea de regresión
        sns.regplot(
            data=subset,
            x="CSE",
            y="Scanned_Path_per_time_per_Block",
            scatter=False,
            ax=ax,
            line_kws={"linewidth": 2}
        )

# Etiquetas y título
ax.set_title("Correlation between CSE and Gaze scanned path", fontsize=16)
ax.set_ylabel("Gaze scanned path", fontsize=14)
ax.set_xlabel("CSE", fontsize=14)

# Ajustar tamaño de la leyenda dentro del gráfico
legend = ax.legend(title="Group", fontsize=12, title_fontsize=14, loc="best", frameon=False)
for text in legend.get_texts():
    text.set_fontsize(14)  # Aumentar tamaño de los textos de la leyenda
legend.get_title().set_fontsize(16)  # Aumentar tamaño del título de la leyenda

# Agregar texto con los valores de correlación en la esquina superior izquierda
text_x = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.05  # Ajuste de posición X
text_y = ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1  # Ajuste de posición Y

correlation_text = "\n".join([
    f"{group}: r = {r:.2f}, p = {p:.3f}" for group, (r, p) in correlation_results.items()
])
#ax.text(text_x, text_y, correlation_text, fontsize=14, bbox=dict(facecolor='white', alpha=0.5))

# Ajustar diseño y guardar la figura
plt.tight_layout()
output_file = Output_Dir + "Figura 10.png"
plt.savefig(output_file)
plt.show()