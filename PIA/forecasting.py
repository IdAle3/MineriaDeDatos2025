import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

def crear_carpeta_salida(ruta):
    """Crear carpeta para guardar resultados si no existe"""
    if not os.path.exists(ruta):
        os.makedirs(ruta)

def cargar_datos(ruta_archivo):
    """Cargar y preparar datos temporales"""
    datos = pd.read_csv(ruta_archivo)
    datos["Fecha"] = pd.to_datetime(datos["Fecha"], errors="coerce")
    datos = datos.dropna(subset=["Fecha"])
    datos["Año"] = datos["Fecha"].dt.year
    return datos

def filtrar_ultimos_anios(datos, cantidad_anios=3):
    """Filtrar datos de los últimos años especificados"""
    año_maximo = datos["Año"].max()
    año_minimo = año_maximo - cantidad_anios + 1
    datos_filtrados = datos[(datos["Año"] >= año_minimo) & (datos["Año"] <= año_maximo)]
    print(f"Analizando datos desde {año_minimo} hasta {año_maximo}")
    return datos_filtrados

def analizar_todos_los_tipos(datos_filtrados):
    """Analizar todos los tipos de crimen y mostrar los más prometedores"""
    datos_mensuales = datos_filtrados.copy()
    datos_mensuales["Mes_Año"] = datos_mensuales["Fecha"].dt.to_period("M")
    
    # Obtener frecuencia de cada tipo
    frecuencias = datos_filtrados["Tipo Principal"].value_counts()
    
    resultados = []
    
    print("=== ANÁLISIS DE TODOS LOS TIPOS DE CRIMEN ===")
    
    for tipo_crimen in datos_mensuales["Tipo Principal"].unique():
        datos_tipo = datos_mensuales[datos_mensuales["Tipo Principal"] == tipo_crimen]
        conteo_mensual = datos_tipo.groupby("Mes_Año").size().reset_index(name="cantidad")
        
        if len(conteo_mensual) >= 6:  
            conteo_mensual["numero_mes"] = range(len(conteo_mensual))
            X = sm.add_constant(conteo_mensual["numero_mes"])
            modelo = sm.OLS(conteo_mensual["cantidad"], X).fit()
            
            # Calcular métricas
            frecuencia_total = frecuencias.get(tipo_crimen, 0)
            promedio_mensual = conteo_mensual["cantidad"].mean()
            desviacion_std = conteo_mensual["cantidad"].std()
            
            if modelo.pvalues.iloc[1] < 0.2:  
                score = (modelo.rsquared * 100) + (promedio_mensual / 10)
                
                resultados.append({
                    'tipo': tipo_crimen,
                    'r_cuadrado': modelo.rsquared,
                    'pendiente': modelo.params.iloc[1],
                    'p_value': modelo.pvalues.iloc[1],
                    'frecuencia_total': frecuencia_total,
                    'promedio_mensual': promedio_mensual,
                    'desviacion_std': desviacion_std,
                    'score': score
                })
    
    # Ordenar por score
    resultados.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n=== TOP 5 TIPOS PARA PRONÓSTICO ===")
    print("Rank | Tipo | R² | Pendiente | Promedio Mensual | Frecuencia Total")
    print("-" * 80)
    
    for i, resultado in enumerate(resultados[:10], 1):
        tendencia = "↑" if resultado['pendiente'] > 0 else "↓"
        print(f"{i:2d}. {resultado['tipo'][:20]:20} | {resultado['r_cuadrado']:5.3f} | {resultado['pendiente']:8.3f}{tendencia} | {resultado['promedio_mensual']:6.1f} | {resultado['frecuencia_total']:4d}")
    
    return resultados

def entrenar_modelo_pronostico(datos_tipo, tipo_crimen):
    """Entrenar modelo para pronóstico mensual"""
    datos_tipo = datos_tipo.copy()
    datos_tipo["Mes_Año"] = datos_tipo["Fecha"].dt.to_period("M").dt.to_timestamp()
    datos_mensuales = datos_tipo.groupby("Mes_Año").size().reset_index(name="numero_crimenes")
    
    if len(datos_mensuales) < 6:
        print(f"Advertencia: Solo hay {len(datos_mensuales)} meses de datos para {tipo_crimen}")
        return None, None
    
    datos_mensuales = datos_mensuales.sort_values("Mes_Año")
    datos_mensuales["meses_desde_inicio"] = (
        (datos_mensuales["Mes_Año"].dt.year - datos_mensuales["Mes_Año"].dt.year.min()) * 12 +
        (datos_mensuales["Mes_Año"].dt.month - datos_mensuales["Mes_Año"].dt.month.min())
    )
    
    X = sm.add_constant(datos_mensuales["meses_desde_inicio"])
    y = datos_mensuales["numero_crimenes"]
    
    modelo = sm.OLS(y, X).fit()
    return modelo, datos_mensuales

def predecir_futuro(modelo, datos_mensuales, meses_prediccion=12):
    """Generar predicciones para meses futuros"""
    ultimo_mes = datos_mensuales["meses_desde_inicio"].max()
    futuro_X = np.arange(ultimo_mes + 1, ultimo_mes + 1 + meses_prediccion)
    
    ultima_fecha = datos_mensuales["Mes_Año"].max()
    fechas_futuras = [ultima_fecha + pd.DateOffset(months=i) for i in range(1, meses_prediccion + 1)]
    
    futuro_X_const = sm.add_constant(futuro_X)
    predicciones = modelo.predict(futuro_X_const)
    
    predicciones = np.maximum(predicciones, 0)
    
    return pd.DataFrame({
        "Fecha": fechas_futuras, 
        "prediccion_crimenes": predicciones
    })

def graficar_pronostico(datos_mensuales, modelo, datos_futuro, tipo_crimen, r_cuadrado):
    """Generar gráfica completa de pronóstico"""
    plt.figure(figsize=(14, 8))
    plt.grid(True, linestyle="--", alpha=0.4)

    # Datos reales
    plt.scatter(
        datos_mensuales["Mes_Año"],
        datos_mensuales["numero_crimenes"],
        color="steelblue",
        label="Datos reales",
        alpha=0.7,
        s=60
    )

    # Línea de regresión
    X_const = sm.add_constant(datos_mensuales["meses_desde_inicio"])
    predicciones = modelo.get_prediction(X_const)
    marco_predicciones = predicciones.summary_frame(alpha=0.05)

    plt.plot(
        datos_mensuales["Mes_Año"],
        marco_predicciones["mean"],
        color="red",
        linewidth=3,
        label="Tendencia lineal"
    )

    plt.fill_between(
        datos_mensuales["Mes_Año"],
        marco_predicciones["mean_ci_lower"],
        marco_predicciones["mean_ci_upper"],
        color="#FFE5B4",
        alpha=0.6,
        label="Intervalo de confianza 95%"
    )

    plt.plot(
        datos_futuro["Fecha"],
        datos_futuro["prediccion_crimenes"],
        color="orange",
        linestyle="--",
        linewidth=3,
        label="Pronóstico futuro"
    )

    ultima_fecha_real = datos_mensuales["Mes_Año"].max()
    plt.axvline(x=ultima_fecha_real, color='gray', linestyle=':', alpha=0.8, linewidth=2)

    plt.title(f"Pronóstico de {tipo_crimen}\nR² = {r_cuadrado:.3f}", fontsize=14, pad=20)
    plt.xlabel("Fecha (agrupación mensual)", fontsize=12)
    plt.ylabel("Cantidad de Crímenes por Mes", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(fontsize=10)
    plt.tight_layout()

    directorio_salida = "resultados_pronostico"
    crear_carpeta_salida(directorio_salida)
    ruta_guardado = f"{directorio_salida}/pronostico_{tipo_crimen.replace('/', '_')}.png"
    plt.savefig(ruta_guardado, dpi=300, bbox_inches="tight")
    plt.close()

def main():
    """Función principal"""
    datos = cargar_datos("Crimenes_Chicago_Limpio_7000.csv")
    
    datos_filtrados = filtrar_ultimos_anios(datos, cantidad_anios=3)
    
    resultados = analizar_todos_los_tipos(datos_filtrados)
    
    if not resultados:
        print("No se encontraron tipos con tendencias significativas.")
        return
    
    mejores_tipos = [r['tipo'] for r in resultados[:5]]
    
    print(f"\n=== SELECCIÓN MANUAL ===")
    print("Tipos recomendados para análisis:")
    for i, tipo in enumerate(mejores_tipos, 1):
        print(f"{i}. {tipo}")
    
    tipos_a_probar = mejores_tipos[:3]
    
    for tipo_crimen in tipos_a_probar:
        print(f"\n{'='*50}")
        print(f"ANALIZANDO: {tipo_crimen}")
        print(f"{'='*50}")
        
        datos_tipo = datos_filtrados[datos_filtrados["Tipo Principal"] == tipo_crimen]
        
        if len(datos_tipo) == 0:
            print(f"No hay datos para {tipo_crimen}")
            continue
            
        print(f"Total de crímenes de este tipo: {len(datos_tipo)}")
        
        modelo, datos_mensuales = entrenar_modelo_pronostico(datos_tipo, tipo_crimen)
        
        if modelo is None:
            print(f"No se pudo entrenar modelo para {tipo_crimen}")
            continue
        
        print("\n=== RESULTADOS DEL MODELO ===")
        print(f"Coeficiente de determinación R²: {modelo.rsquared:.4f}")
        print(f"Pendiente de la tendencia: {modelo.params.iloc[1]:.4f}")
        print(f"Significancia estadística (p-value): {modelo.pvalues.iloc[1]:.4f}")
        
        if modelo.rsquared > 0.3 and modelo.pvalues.iloc[1] < 0.1:
            datos_futuro = predecir_futuro(modelo, datos_mensuales, meses_prediccion=12)
            print("\nPronóstico para los próximos 6 meses:")
            print(datos_futuro[["Fecha", "prediccion_crimenes"]].head(6).round(1))
            
            # Estadísticas descriptivas
            promedio_historico = datos_mensuales["numero_crimenes"].mean()
            max_historico = datos_mensuales["numero_crimenes"].max()
            min_historico = datos_mensuales["numero_crimenes"].min()
            
            print(f"\nEstadísticas históricas:")
            print(f"Promedio mensual: {promedio_historico:.1f}")
            print(f"Máximo mensual: {max_historico:.1f}")
            print(f"Mínimo mensual: {min_historico:.1f}")
            
            graficar_pronostico(datos_mensuales, modelo, datos_futuro, tipo_crimen, modelo.rsquared)
            print(f"\n✓ Gráfica de pronóstico generada: {tipo_crimen}")
            
            if tipo_crimen != tipos_a_probar[-1]:
                continuar = input("\n¿Continuar con el siguiente tipo? (s/n): ")
                if continuar.lower() != 's':
                    break
        else:
            print("El modelo no es suficientemente predictivo (R² < 0.3 o p-value > 0.1)")

if __name__ == "__main__":
    main()