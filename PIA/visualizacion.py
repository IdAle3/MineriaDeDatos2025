import pandas as pd
import matplotlib.pyplot as plt
import os

def crear_carpeta_salida(ruta):
    """Crear carpeta para guardar resultados si no existe"""
    if not os.path.exists(ruta):
        os.makedirs(ruta)

def cargar_datos(ruta_archivo):
    """Cargar y preparar los datos"""
    datos = pd.read_csv(ruta_archivo)
    
    # Ajustar tipos de datos para números enteros
    columnas_enteras = ["Año", "Mes", "Día", "Hora", "Distrito"]
    for columna in columnas_enteras:
        if columna in datos.columns:
            datos[columna] = datos[columna].astype("Int64")
    
    return datos

def generar_grafica_torta_tipos_crimen(datos, carpeta_salida):
    """Generar gráfico de torta con los 5 tipos de crimen más comunes"""
    plt.figure(figsize=(8, 8))
    datos["Tipo Principal"].value_counts().head(5).plot.pie(
        autopct="%1.1f%%", startangle=90, cmap="Set3"
    )
    plt.title("Distribución de los 5 Tipos de Crimen Más Comunes en Chicago")
    plt.ylabel("")
    plt.savefig(f"{carpeta_salida}/distribucion_tipos_crimen.png")
    plt.close()

def generar_histograma_horas(datos, carpeta_salida):
    """Generar histograma de distribución por horas"""
    plt.figure(figsize=(10, 6))
    datos["Hora"].hist(bins=24, color="skyblue", edgecolor="black")
    plt.title("Distribución de Crímenes por Hora del Día")
    plt.xlabel("Hora del Día")
    plt.ylabel("Cantidad de Crímenes")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{carpeta_salida}/distribucion_horas.png")
    plt.close()

def generar_mapa_calor_geografico(datos, carpeta_salida):
    """Generar mapa de dispersión geográfica"""
    plt.figure(figsize=(10, 8))
    plt.scatter(datos["Longitud"], datos["Latitud"], alpha=0.1, color="purple")
    plt.title("Mapa de Densidad de Crímenes en Chicago")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.savefig(f"{carpeta_salida}/mapa_geografico.png")
    plt.close()

def generar_top_distritos(datos, carpeta_salida):
    """Generar gráfico de barras de los distritos con más crímenes"""
    plt.figure(figsize=(12, 6))
    datos["Distrito"].value_counts().head(10).plot(
        kind="bar", color="orange", edgecolor="black"
    )
    plt.title("Top 10 Distritos con Mayor Cantidad de Crímenes")
    plt.xlabel("Número de Distrito")
    plt.ylabel("Cantidad de Crímenes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{carpeta_salida}/top_distritos.png")
    plt.close()

def main():
    """Función principal"""
    carpeta_salida = "resultados_visualizacion"
    crear_carpeta_salida(carpeta_salida)
    
    datos = cargar_datos("Crimenes_Chicago_Limpio_7000.csv")
    
    
    # Generar todas las gráficas
    generar_grafica_torta_tipos_crimen(datos, carpeta_salida)
    generar_histograma_horas(datos, carpeta_salida)
    generar_mapa_calor_geografico(datos, carpeta_salida)
    generar_top_distritos(datos, carpeta_salida)
    
    print(f"Visualizaciones guardadas en la carpeta: {carpeta_salida}")

if __name__ == "__main__":
    main()