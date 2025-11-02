import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# Cargar y preparar datos

def cargar_datos(ruta):
    df = pd.read_csv(ruta)
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha"])
    df["Año"] = df["Fecha"].dt.year
    return df

# Filtrar últimos N años

def filtrar_ultimos_anios(df, n=3):
    año_max = df["Año"].max()
    año_min = año_max - n + 1
    df_filtrado = df[(df["Año"] >= año_min) & (df["Año"] <= año_max)]
    print(f"Usando datos desde {año_min} hasta {año_max}")
    return df_filtrado

# Encontrar tipo con mayor tendencia

def tipo_con_mayor_tendencia(df_filtrado):
    df_mensual = df_filtrado.copy()
    df_mensual["Mes_Año"] = df_mensual["Fecha"].dt.to_period("M")
    
    tendencias = []
    
    for tipo in df_mensual["Tipo Principal"].unique():
        datos_tipo = df_mensual[df_mensual["Tipo Principal"] == tipo]
        conteo_mensual = datos_tipo.groupby("Mes_Año").size().reset_index(name="conteo")
        
        if len(conteo_mensual) >= 6:
            conteo_mensual["mes_num"] = range(len(conteo_mensual))
            X = sm.add_constant(conteo_mensual["mes_num"])
            modelo = sm.OLS(conteo_mensual["conteo"], X).fit()
            
            if modelo.pvalues.iloc[1] < 0.1:
                score_tendencia = abs(modelo.params.iloc[1]) * modelo.rsquared
                tendencias.append((tipo, modelo.params.iloc[1], modelo.rsquared, score_tendencia))
    
    if not tendencias:
        tipo_top = df_mensual["Tipo Principal"].mode()[0]
        print(f"Usando tipo más frecuente: {tipo_top}")
        return tipo_top
    
    tendencias.sort(key=lambda x: x[3], reverse=True)
    tipo_top = tendencias[0][0]
    
    print(f"Tipo con mayor tendencia: {tipo_top}")
    return tipo_top

# Entrenar modelo mensual
def entrenar_modelo_mensual(df_tipo):
    df_tipo = df_tipo.copy()
    df_tipo["Mes_Año"] = df_tipo["Fecha"].dt.to_period("M").dt.to_timestamp()
    df_by_month = df_tipo.groupby("Mes_Año").size().reset_index(name="num_crimenes")
    
    df_by_month = df_by_month.sort_values("Mes_Año")
    df_by_month["months_since_start"] = (
        (df_by_month["Mes_Año"].dt.year - df_by_month["Mes_Año"].dt.year.min()) * 12 +
        (df_by_month["Mes_Año"].dt.month - df_by_month["Mes_Año"].dt.month.min())
    )
    
    X = sm.add_constant(df_by_month["months_since_start"])
    y = df_by_month["num_crimenes"]
    
    modelo = sm.OLS(y, X).fit()
    return modelo, df_by_month

#  Predecir meses futuros

def predecir_futuro(modelo, df_by_month, meses=12):
    last_month = df_by_month["months_since_start"].max()
    future_X = np.arange(last_month + 1, last_month + 1 + meses)
    
    last_date = df_by_month["Mes_Año"].max()
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, meses + 1)]
    
    future_X_const = sm.add_constant(future_X)
    predicciones = modelo.predict(future_X_const)
    
    return pd.DataFrame({
        "Fecha": future_dates, 
        "num_crimenes_pred": predicciones
    })

# Graficar resultados
def graficar(df_by_month, modelo, future_df, tipo):
    plt.figure(figsize=(12, 6))
    plt.grid(True, linestyle="--", alpha=0.4)

    # Puntos reales
    plt.scatter(
        df_by_month["Mes_Año"],
        df_by_month["num_crimenes"],
        color="steelblue",
        label="Datos reales",
        alpha=0.7
    )

    # Línea de regresión
    X_const = sm.add_constant(df_by_month["months_since_start"])
    predicciones = modelo.get_prediction(X_const)
    pred_frame = predicciones.summary_frame(alpha=0.05)

    plt.plot(
        df_by_month["Mes_Año"],
        pred_frame["mean"],
        color="red",
        linewidth=2,
        label="Regresión lineal"
    )

    # Intervalo de confianza
    plt.fill_between(
        df_by_month["Mes_Año"],
        pred_frame["mean_ci_lower"],
        pred_frame["mean_ci_upper"],
        color="#FFE5B4",
        alpha=0.6,
        label="Intervalo de confianza 95%"
    )

    # Predicción futura
    plt.plot(
        future_df["Fecha"],
        future_df["num_crimenes_pred"],
        color="orange",
        linestyle="--",
        linewidth=2,
        label="Predicción futura"
    )

    # Línea separadora
    ultima_fecha_real = df_by_month["Mes_Año"].max()
    plt.axvline(x=ultima_fecha_real, color='gray', linestyle=':', alpha=0.8)

    plt.title(f"Forecasting de Crímenes - {tipo}")
    plt.xlabel("Fecha (mensual)")
    plt.ylabel("Número de crímenes por mes")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    os.makedirs("Practica8/img", exist_ok=True)
    ruta = f"Practica8/img/forecast_{tipo.replace('/', '_')}.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df = cargar_datos("Practica1/Crimenes_Chicago_Limpio_7000.csv")
    
    df_filtrado = filtrar_ultimos_anios(df, n=3)
    
  
    tipo_top = tipo_con_mayor_tendencia(df_filtrado)
    df_tipo = df_filtrado[df_filtrado["Tipo Principal"] == tipo_top]
    
    modelo, df_by_month = entrenar_modelo_mensual(df_tipo)
    
    print("\nResultados del modelo:")
    print(f"R²: {modelo.rsquared:.4f}")
    print(f"Coeficiente (pendiente): {modelo.params.iloc[1]:.4f}")
    print(f"P-value pendiente: {modelo.pvalues.iloc[1]:.4f}")
    
    if modelo.rsquared > 0.1 and modelo.pvalues.iloc[1] < 0.1:
        future_df = predecir_futuro(modelo, df_by_month, meses=12)
        print("Predicciones proximos 5 meses:")
        print(future_df[["Fecha", "num_crimenes_pred"]].head().round(2))
        
        graficar(df_by_month, modelo, future_df, tipo_top)
    
if __name__ == "__main__":
    main()