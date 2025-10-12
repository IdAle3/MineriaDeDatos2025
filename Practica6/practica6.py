# Práctica 6 - KNN y análisis de crímenes en Chicago 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


OUTPUT_DIR = "img"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Cargar el dataset 
df = pd.read_csv("../Practica1/Crimenes_Chicago_Limpio_7000.csv")


# Función get_cmap 
def get_cmap(n, name="tab10"):
    cmap = plt.colormaps.get_cmap(name)
    return [cmap(i / n) for i in range(n)]

# Gráfica general agrupada por tipo de crimen 
def scatter_group_by(file_path: str, df: pd.DataFrame, x_column: str, y_column: str, label_column: str):
    labels = df[label_column].unique()
    colors = get_cmap(len(labels))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(labels):
        subset = df[df[label_column] == label]
        ax.scatter(subset[x_column], subset[y_column], label=label, alpha=0.5, s=25, color=colors[i])
    ax.legend(fontsize="small", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title(f"Crímenes en Chicago por {label_column}")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

scatter_group_by(f"{OUTPUT_DIR}/crimenes_por_tipo.png", df, "Longitud", "Latitud", "Tipo Principal")

# Implementación de KNN 
def k_nearest_neighbors(points, labels, new_points, k=5):
    predictions = []
    for new_point in new_points:
        distances = np.linalg.norm(points - new_point, axis=1)
        nearest_indices = distances.argsort()[:k]
        nearest_labels = labels[nearest_indices]

        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        majority = unique_labels[np.argmax(counts)]

        predictions.append(majority)
    return np.array(predictions)

#  de KNN con los primeros 100 registros 
points = df[["Longitud", "Latitud"]].values[:100]
labels = df["Tipo Principal"].values[:100]
new_points = np.array([[-87.65, 41.85], [-87.70, 41.90]])
predictions = k_nearest_neighbors(points, labels, new_points, k=5)

# Para predecir los dos tipo de crimen más probable (usando KNN) en nuevas ubicaciones
# basándose en crímenes cercanos geográficamente. 

print("Predicciones para los nuevos puntos:")
for i, pred in enumerate(predictions):
    print(f"  Punto {i+1}: {pred}")

# Análisis temporal: año y mes más violentos. Se realizó un análisis temporal para reducir la 
# saturación visual del mapa general y facilitar la interpretación de los datos. Por eso saqué 
# los top 5 crímenes que más se cometieon del mes mas violento del año más violento.

crimenes_por_año = df["Año"].value_counts().sort_index(ascending=False)
año_mas_violento = crimenes_por_año.idxmax()

df_año = df[df["Año"] == año_mas_violento]
crimenes_por_mes = df_año["Mes"].value_counts().sort_index()
mes_mas_violento = crimenes_por_mes.idxmax()

df_mes = df_año[df_año["Mes"] == mes_mas_violento]

print(f"\nAño más violento: {año_mas_violento}")
print(f"Mes más violento: {mes_mas_violento} ({len(df_mes)} crímenes)")

# Top 5 crímenes del mes más violento 
def scatter_top_groups_month(file_path: str, df: pd.DataFrame, x_column: str, y_column: str, label_column: str, top_n=5):
    top_labels = df[label_column].value_counts().nlargest(top_n).index
    df_top = df[df[label_column].isin(top_labels)]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = get_cmap(len(top_labels))
    for i, label in enumerate(top_labels):
        subset = df_top[df_top[label_column] == label]
        ax.scatter(subset[x_column], subset[y_column], label=label, alpha=0.6, s=25, color=colors[i])
    ax.legend(title=f"Top {top_n} crímenes", fontsize="small", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title(f"Top {top_n} crímenes - {año_mas_violento}/{mes_mas_violento}")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

scatter_top_groups_month(f"{OUTPUT_DIR}/crimenes_top5_mes_mas_violento.png", df_mes, "Longitud", "Latitud", "Tipo Principal", top_n=5)

# Tipo de crimen más frecuente ese mes 
top_crimen = df_mes["Tipo Principal"].value_counts().idxmax()
porcentaje = round((df_mes["Tipo Principal"].value_counts().max() / len(df_mes)) * 100, 2)
print(f"Crimen más frecuente en {año_mas_violento}-{mes_mas_violento}: {top_crimen} ({porcentaje}%)")

