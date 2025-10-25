import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import StandardScaler


np.random.seed(42)  # Fija la semilla para resultados consistentes

# FUNCIONES BASE

def euclidean_distance(p_1: np.array, p_2: np.array) -> float:
    return np.sqrt(np.sum((p_2 - p_1) ** 2))

def calculate_means(points: np.array, labels: np.array, clusters: int) -> np.array:
    means = []
    for k in range(clusters):
        cluster_points = points[labels == k]
        if len(cluster_points) > 0:
            m = np.mean(cluster_points, axis=0)
        else:
            m = np.zeros(points.shape[1])
        means.append(m)
    return np.array(means)

def calculate_nearest_k(point: np.array, actual_means: np.array) -> int:
    distance = [euclidean_distance(mean, point) for mean in actual_means]
    return np.argmin(distance)

def scatter_group_by(file_path: str, df: pd.DataFrame, x_column: str, y_column: str, label_column: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    labels = pd.unique(df[label_column])
    for label in labels:
        fdf = df[df[label_column] == label]
        s = 90 if label == "centroid" else 10  
        ax.scatter(fdf[x_column], fdf[y_column], label=label, s=s)
    
    ax.set_xlabel("Longitud (x)")
    ax.set_ylabel("Latitud (y)")
    ax.set_title(os.path.basename(file_path))
    ax.invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def k_means(points: np.array, k: int, img_path: str, max_iter: int = 15):
    N = len(points)
    y = np.random.randint(0, k, N)  
    for t in range(max_iter):
        actual_means = calculate_means(points, y, k)
        new_y = np.array([calculate_nearest_k(p, actual_means) for p in points])

        # Preparar DataFrame para graficar
        df_points = pd.DataFrame(points, columns=['x', 'y'])
        df_points['label'] = np.char.mod('%d', new_y)
        df_means = pd.DataFrame(actual_means, columns=['x', 'y'])
        df_means['label'] = ['centroid'] * k
        df_all = pd.concat([df_points, df_means])

        # Guardar gráfica de esta iteración
        scatter_group_by(os.path.join(img_path, f"kmeans_{t}.png"), df_all, "x", "y", "label")


        if np.array_equal(new_y, y):
            print(f" K-Means convergió en {t+1} iteraciones")
            break
        y = new_y.copy()

    return actual_means, y

# CARGA DE DATOS

df = pd.read_csv("Practica1/Crimenes_Chicago_Limpio_7000.csv")

df['Violencia Doméstica'] = df['Violencia Doméstica'].astype(str).str.strip().str.lower()
violentos = df[df['Violencia Doméstica'].isin(['sí', 'si', 'true', 'yes'])]

año_mas_violento = violentos['Año'].value_counts().idxmax()
subset = violentos[violentos['Año'] == año_mas_violento]

coords = subset[['Longitud', 'Latitud']].dropna().to_numpy()

# Normalizar coordenadas
coords = StandardScaler().fit_transform(coords)


img_path = "Practica7/img"
os.makedirs(img_path, exist_ok=True)


# EJECUTAR K-MEANS
means, labels = k_means(coords, k=4, img_path=img_path, max_iter=15)


# GRÁFICA FINAL
df_plot = pd.DataFrame(coords, columns=['x', 'y'])
df_plot['label'] = np.char.mod('%d', labels)
df_centroids = pd.DataFrame(means, columns=['x', 'y'])
df_centroids['label'] = ['centroid'] * len(means)
final_df = pd.concat([df_plot, df_centroids])

scatter_group_by(os.path.join(img_path, "kmeans_final.png"), final_df, "x", "y", "label")

print("\n Centroides finales encontrados:")
print(means)
print(f"\nImágenes guardadas en: {img_path}")
