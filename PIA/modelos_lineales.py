import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import numpy as np

def crear_carpeta_salida(ruta):
    """Crear carpeta para guardar resultados si no existe"""
    if not os.path.exists(ruta):
        os.makedirs(ruta)

def cargar_datos(ruta_archivo):
    """Cargar y preparar los datos"""
    return pd.read_csv(ruta_archivo)

def obtener_anio_mas_crimenes(datos):
    """Identificar el año con mayor cantidad de crímenes"""
    crimenes_por_anio = datos["Año"].value_counts().sort_index()
    anio_maximo = crimenes_por_anio.idxmax()
    return anio_maximo

def agrupar_por_variable(datos, anio, variable):
    """Agrupar crímenes por variable específica"""
    datos_anio = datos[datos["Año"] == anio]
    crimenes_agrupados = datos_anio.groupby(variable).size().reset_index(
        name="Cantidad de Crímenes"
    )
    return crimenes_agrupados

def entrenar_modelo_regresion(X, y):
    """Entrenar modelo de regresión lineal"""
    modelo = LinearRegression()
    modelo.fit(X, y)
    predicciones = modelo.predict(X)
    r_cuadrado = modelo.score(X, y)
    
    print(f"Pendiente: {modelo.coef_[0]:.4f}")
    print(f"Intersección: {modelo.intercept_:.4f}")
    print(f"Coeficiente de determinación (R²): {r_cuadrado:.4f}")
    
    return modelo, predicciones, r_cuadrado

def graficar_regresion(X, y, predicciones, anio, variable, r_cuadrado, ruta_guardado):
    """Generar gráfica de regresión lineal"""
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color="purple", label="Datos reales")
    plt.plot(X, predicciones, color="orange", linewidth=2, label="Línea de regresión")
    plt.title(f"Regresión Lineal - Crímenes por {variable} en {anio}")
    plt.xlabel(variable)
    plt.ylabel("Cantidad de Crímenes")
    plt.text(0.5, max(y)*0.9, f"R² = {r_cuadrado:.3f}", fontsize=12, color="gray")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(ruta_guardado)
    plt.close()
    print(f"Gráfica guardada en: {ruta_guardado}")

def modelo_crimenes_por_mes(datos, anio, directorio_salida):
    """Modelo de regresión para crímenes por mes"""
    crimenes_mes = agrupar_por_variable(datos, anio, "Mes")
    X = crimenes_mes["Mes"].values.reshape(-1, 1)
    y = crimenes_mes["Cantidad de Crímenes"].values
    
    print("\n=== Modelo: Crímenes por Mes ===")
    modelo, predicciones, r_cuadrado = entrenar_modelo_regresion(X, y)
    
    ruta_guardado = f"{directorio_salida}/regresion_mensual_{anio}.png"
    graficar_regresion(X, y, predicciones, anio, "Mes", r_cuadrado, ruta_guardado)

def modelo_crimenes_por_hora(datos, anio, directorio_salida):
    """Modelo de regresión para crímenes por hora"""
    crimenes_hora = agrupar_por_variable(datos, anio, "Hora")
    X = crimenes_hora["Hora"].values.reshape(-1, 1)
    y = crimenes_hora["Cantidad de Crímenes"].values
    
    print("\n=== Modelo: Crímenes por Hora ===")
    modelo, predicciones, r_cuadrado = entrenar_modelo_regresion(X, y)
    
    ruta_guardado = f"{directorio_salida}/regresion_horaria_{anio}.png"
    graficar_regresion(X, y, predicciones, anio, "Hora", r_cuadrado, ruta_guardado)

def main():
    """Función principal"""
    directorio_salida = "resultados_modelos"
    crear_carpeta_salida(directorio_salida)
    
    datos = cargar_datos("Crimenes_Chicago_Limpio_7000.csv")
    anio_maximo = obtener_anio_mas_crimenes(datos)
    
    print(f"Año con mayor criminalidad: {anio_maximo}")
    
    # Ejecutar ambos modelos
    modelo_crimenes_por_mes(datos, anio_maximo, directorio_salida)
    modelo_crimenes_por_hora(datos, anio_maximo, directorio_salida)

if __name__ == "__main__":
    main()