import re
import os
import pandas as pd

# --- Función para calcular moda ---
def obtener_moda(serie):
    modos = serie.mode()
    if len(modos) == 0 or len(modos) == len(serie):
        return "Sin moda"
    return modos.iloc[0]

# --- Agrupación flexible ---
def analizar_agrupado(df, agrupacion):
    if isinstance(agrupacion, str):
        agrupacion = [agrupacion]

    # Variables categóricas
    cat_vars = ["Año", "Mes", "Día"]
    # Variable numérica
    num_vars = ["Hora"]

    # Diccionario de agregaciones
    agg_dict = {}

    for var in cat_vars:
        if var in df.columns:
            agg_dict[var] = ["count", "min", "max", obtener_moda]

    for var in num_vars:
        if var in df.columns:
            agg_dict[var] = ["count", "mean", "min", "max", "var", "std", obtener_moda, pd.Series.kurt]

    df_grouped = df.groupby(agrupacion).agg(agg_dict).reset_index()

    print(f"\nESTADÍSTICAS AGRUPADAS POR: {agrupacion}")
    print(df_grouped.head())

    guardar_csv(df_grouped, agrupacion)

# --- Guardar resultados ---
def guardar_csv(df_grouped, agrupacion):
    
    nombre = "_".join(agrupacion)
    #Quitar caracteres inválidos para Windows
    nombre = re.sub(r'[\\/*?:"<>|¿?]', "", nombre).replace(" ", "_")
    
    ruta = os.path.join(f'estadisticas_{nombre}.csv')
    df_grouped.to_csv(ruta, index=False)


# --- Análisis general sin agrupación ---
def analizar_general(df):
    resultados = {}

    for var in ["Año", "Mes", "Día"]:
        resultados[var] = {
            "conteo": df[var].count(),
            "mínimo": df[var].min(),
            "máximo": df[var].max(),
            "moda": obtener_moda(df[var])
        }

    resultados["Hora"] = {
        "conteo": df["Hora"].count(),
        "media": df["Hora"].mean(),
        "mínimo": df["Hora"].min(),
        "máximo": df["Hora"].max(),
        "varianza": df["Hora"].var(),
        "desv_est": df["Hora"].std(),
        "moda": obtener_moda(df["Hora"]),
        "curtosis": df["Hora"].kurt()
    }

    print("\nESTADÍSTICAS GENERALES DEL DATASET:")
    print(pd.DataFrame(resultados))

# --- Main ---
def main():
    direccion_actual = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(direccion_actual, 'Crimenes_Chicago_Limpio_7000.csv'))

    # Agrupaciones pedidas
    analizar_agrupado(df, "Tipo Principal")
    analizar_agrupado(df, ["Tipo Principal", "¿Hubo Arresto?"])
    analizar_agrupado(df, "Violencia Doméstica")
    analizar_agrupado(df, "Ubicación")
    analizar_agrupado(df, "Año")

    # Estadísticas globales
    analizar_general(df)

if __name__ == "__main__":
    main()
