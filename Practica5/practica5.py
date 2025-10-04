# === Práctica 5 - Regresiones Lineales con Funciones ===
# Modelos: 
# Número de crímenes por mes 
# Número de crímenes por hora 

import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import numpy as np

# === 1. Crear carpeta de salida ===
def crear_carpeta_salida(ruta):
    if not os.path.exists(ruta):
        os.makedirs(ruta)

# === 2. Cargar datos ===
def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    return df


# === 3. Obtener el año con más crímenes ===
def obtener_anio_max(df):
    crimenes_por_anio = df["Año"].value_counts().sort_index()
    anio_max = crimenes_por_anio.idxmax()
    return anio_max

# === 4. Agrupar crímenes por variable (Mes u Hora) ===
def agrupar_por_variable(df, anio, variable):
    df_anio = df[df["Año"] == anio]
    crimenes_por_var = df_anio.groupby(variable).size().reset_index(name="Número de Crímenes")
    return crimenes_por_var

# === 5. Entrenar modelo de regresión ===
def entrenar_modelo(X, y):
    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)
    r2 = modelo.score(X, y)
    print(f"Pendiente: {modelo.coef_[0]:.4f}")
    print(f"Intersección: {modelo.intercept_:.4f}")
    print(f"Coeficiente de determinación (R²): {r2:.4f}")
    return modelo, y_pred, r2

# === 6. Graficar resultados ===
def graficar_regresion(X, y, y_pred, anio, variable, r2, salida):
    plt.figure(figsize=(8,5))
    plt.scatter(X, y, color="purple", label="Datos reales")
    plt.plot(X, y_pred, color="orange", linewidth=2, label="Línea de regresión")
    plt.title(f"Regresión Lineal - Crímenes por {variable} en {anio}")
    plt.xlabel(variable)
    plt.ylabel("Número de Crímenes")
    plt.text(0.5, max(y)*0.9, f"R² = {r2:.3f}", fontsize=10, color="gray")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(salida)
    plt.close()
    print(f" Gráfica guardada en: {salida}")

# === 7. Modelo 1: Crímenes por mes ===
def modelo_crimenes_por_mes(df, anio, output_dir):
    crimenes_mes = agrupar_por_variable(df, anio, "Mes")
    X = crimenes_mes["Mes"].values.reshape(-1, 1)
    y = crimenes_mes["Número de Crímenes"].values
    print("\n=== Modelo 1: Crímenes por mes ===")
    modelo, y_pred, r2 = entrenar_modelo(X, y)
    salida = f"{output_dir}/regresion_crimenes_mes_{anio}.png"
    graficar_regresion(X, y, y_pred, anio, "Mes", r2, salida)

# === 8. Modelo 2: Crímenes por hora ===
def modelo_crimenes_por_hora(df, anio, output_dir):
    crimenes_hora = agrupar_por_variable(df, anio, "Hora")
    X = crimenes_hora["Hora"].values.reshape(-1, 1)
    y = crimenes_hora["Número de Crímenes"].values
    print("\n=== Modelo 2: Crímenes por hora ===")
    modelo, y_pred, r2 = entrenar_modelo(X, y)
    salida = f"{output_dir}/regresion_crimenes_hora_{anio}.png"
    graficar_regresion(X, y, y_pred, anio, "Hora", r2, salida)

# === 9. Función principal ===
def main():
    DATA_PATH = "../Practica1/Crimenes_Chicago_Limpio_7000.csv"
    OUTPUT_DIR = "img"
    crear_carpeta_salida(OUTPUT_DIR)

    df = cargar_datos(DATA_PATH)
    if df is None:
        return

    anio_max = obtener_anio_max(df)

    # Ejecutar ambos modelos
    modelo_crimenes_por_mes(df, anio_max, OUTPUT_DIR)
    modelo_crimenes_por_hora(df, anio_max, OUTPUT_DIR)

# === 10. Punto de entrada ===
if __name__ == "__main__":
    main()
