# Práctica 3 - Data Visualization
# Graficando crímenes de Chicago con funciones

import pandas as pd
import matplotlib.pyplot as plt
import os

# === 1. Crear carpeta "graficas" dentro de Practica3 si no existe ===
OUTPUT_DIR = "graficas"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# === 2. Cargar dataset limpio desde Practica1 ===
df = pd.read_csv("../Practica1/Crimenes_Chicago_Limpio_7000.csv")

# === 3. Ajustar tipos de datos para evitar decimales innecesarios ===
cols_enteras = ["Año", "Mes", "Día", "Hora", "Distrito", "Barrio", "Área Comunitaria"]
for col in cols_enteras:
    if col in df.columns:
        df[col] = df[col].astype("Int64")  # Soporta nulos pero sin decimales


# ==============================
# Funciones de gráficas
# ==============================

def graficar_pie_tipos(df):
    plt.figure(figsize=(6,6))
    df["Tipo Principal"].value_counts().head(5).plot.pie(
        autopct="%1.1f%%", startangle=90, cmap="Set3"
    )
    plt.title("Top 5 Tipos de Crímenes en Chicago")
    plt.ylabel("")
    plt.savefig(f"{OUTPUT_DIR}/pie_tipos.png")
    plt.close()


def graficar_histograma_horas(df):
    plt.figure(figsize=(8,6))
    df["Hora"].hist(bins=24, color="skyblue", edgecolor="black")
    plt.title("Distribución de Crímenes por Hora del Día")
    plt.xlabel("Hora")
    plt.ylabel("Frecuencia")
    plt.savefig(f"{OUTPUT_DIR}/histograma_horas.png")
    plt.close()


def graficar_boxplot_distrito(df):
    plt.figure(figsize=(12,6))
    plt.boxplot(
        [df[df["Distrito"] == d].groupby("Tipo Principal").size().values 
         for d in df["Distrito"].dropna().unique()],
        labels=df["Distrito"].dropna().unique()
    )
    plt.title("Distribución de crímenes por distrito")
    plt.xlabel("Distrito")
    plt.ylabel("Número de crímenes por tipo")
    plt.xticks(rotation=90)
    plt.savefig(f"{OUTPUT_DIR}/boxplot_crimenes_por_distrito.png")
    plt.close()


def graficar_barplot_distritos(df):
    top_distritos = df["Distrito"].value_counts().head(10)
    plt.figure(figsize=(10,6))
    top_distritos.plot(kind="bar", color="orange", edgecolor="black")
    plt.title("Top 10 Distritos con más Crímenes")
    plt.xlabel("Distrito")
    plt.ylabel("Número de Crímenes")
    plt.savefig(f"{OUTPUT_DIR}/barplot_distritos.png")
    plt.close()


def graficar_scatter_mapa(df):
    plt.figure(figsize=(8,6))
    plt.scatter(df["Longitud"], df["Latitud"], alpha=0.1, color="purple")
    plt.title("Mapa de Crímenes en Chicago (Latitud vs Longitud)")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.savefig(f"{OUTPUT_DIR}/scatter_mapa.png")
    plt.close()


def graficar_histogramas_numericos(df):
    columnas_numericas = ["Año", "Mes", "Hora", "Distrito"]
    for col in columnas_numericas:
        if col in df.columns:
            plt.figure(figsize=(8,6))
            df[col].hist(bins=20, color="lightgreen", edgecolor="black")
            plt.title(f"Histograma de {col}")
            plt.xlabel(col)
            plt.ylabel("Frecuencia")
            plt.savefig(f"{OUTPUT_DIR}/histograma_{col}.png")
            plt.close()


# ==============================
# Main
# ==============================

def main():
    graficar_pie_tipos(df)
    graficar_histograma_horas(df)
    graficar_boxplot_distrito(df)
    graficar_barplot_distritos(df)
    graficar_scatter_mapa(df)
    graficar_histogramas_numericos(df)
    print(f"Se generaron todas las gráficas en la carpeta: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
