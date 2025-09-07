import os
import pandas as pd

# --- Diccionarios para traducir meses y días ---
meses = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

dias_semana_map = {
    "Monday": "Lunes",
    "Tuesday": "Martes",
    "Wednesday": "Miércoles",
    "Thursday": "Jueves",
    "Friday": "Viernes",
    "Saturday": "Sábado",
    "Sunday": "Domingo"
}

# --- Función para obtener la moda ---
def obtener_moda(serie):
    modos = serie.mode()
    if len(modos) == 0 or len(modos) == len(serie):
        return "No hay moda"
    else:
        return modos.iloc[0]

# --- Función para generar nombres de columnas ---
def nombres_columnas(agg_dict):
    columnas = []
    for col, funcs in agg_dict.items():
        for func in funcs:
            if callable(func):
                columnas.append(f"{col}_{func.__name__}")
            else:
                columnas.append(f"{col}_{func}")
    return columnas

# --- Función para guardar CSV ---
def guardar_csv(df_agrupado, nombre):
    nombre = nombre.replace(" ", "_").replace("?", "")
    df_agrupado.to_csv(f"estadisticas_{nombre}.csv", index=False)

# --- Función de análisis agrupado ---
def analizar_agrupado(df, agrupacion):
    if isinstance(agrupacion, str):
        agrupacion = [agrupacion]

    # Diccionario de funciones por columna
    agg_dict = {
        "Año": ["count", "mean", "min", "max", "var", "std", obtener_moda, pd.Series.kurt],
        "Mes": ["count", "mean", "min", "max", "var", "std", obtener_moda, pd.Series.kurt],
        "Día": ["count", "mean", "min", "max", "var", "std", obtener_moda, pd.Series.kurt],
        "Hora": ["count", "mean", "min", "max", "var", "std", obtener_moda, pd.Series.kurt]
    }

    # Si existe columna de día de la semana
    if "Dia_semana" in df.columns:
        agg_dict["Dia_semana"] = ["count", obtener_moda]

    # Agrupamos y aplicamos funciones
    df_agrupado = df.groupby(agrupacion).agg(agg_dict)
    
    # Resetear el índice para convertir las columnas de agrupación en columnas normales
    df_agrupado = df_agrupado.reset_index()
    
    # Generar nombres de columnas
    nombres_agg = nombres_columnas(agg_dict)
    new_columns = list(agrupacion) + nombres_agg
    
    # Asignar nuevos nombres a las columnas
    df_agrupado.columns = new_columns

    # Traducimos Mes y Día de la semana
    if "Mes_obtener_moda" in df_agrupado.columns:
        df_agrupado["Mes_obtener_moda"] = df_agrupado["Mes_obtener_moda"].map(meses)
    if "Dia_semana_obtener_moda" in df_agrupado.columns:
        df_agrupado["Dia_semana_obtener_moda"] = df_agrupado["Dia_semana_obtener_moda"].map(dias_semana_map)

    print(f"\n📊 ESTADÍSTICAS AGRUPADAS POR: {agrupacion}")
    print(df_agrupado.head())

    guardar_csv(df_agrupado, "_".join(agrupacion))

# --- Función principal ---
def main():
    direccion_actual = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(direccion_actual, "Crimenes_Chicago_Limpio_7000.csv"))

    # Creamos columna de Fecha y Día de la semana
    df["Fecha"] = pd.to_datetime(
        df[["Año","Mes","Día"]].rename(columns={"Año":"year","Mes":"month","Día":"day"}),
        errors='coerce'
    )
    df["Dia_semana"] = df["Fecha"].dt.day_name().map(dias_semana_map)

    # --- Estadísticas agrupadas ---
    analizar_agrupado(df, "Tipo Principal")
    analizar_agrupado(df, ["Tipo Principal", "¿Hubo Arresto?"])
    analizar_agrupado(df, "Violencia Doméstica")
    analizar_agrupado(df, "Ubicación")
    analizar_agrupado(df, "Año")
    analizar_agrupado(df, "Mes")
    analizar_agrupado(df, "Dia_semana")

    # --- Estadísticas generales ---
    cols_num = ["Año","Mes","Día","Hora"]
    df_stats = df[cols_num].agg(["count","mean","min","max","var","std", obtener_moda, pd.Series.kurt]).transpose()
    df_stats = df_stats.rename(columns={
        "count": "Conteo",
        "mean": "Media",
        "min": "Mínimo",
        "max": "Máximo",
        "var": "Varianza",
        "std": "DesvEst",
        "obtener_moda": "Moda",
        "kurt": "Curtosis"
    })
    print("\n📊 ESTADÍSTICAS GENERALES DEL DATASET:")
    print(df_stats)

if __name__ == "__main__":
    main()