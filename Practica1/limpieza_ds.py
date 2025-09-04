# Práctica 1 - Data Cleaning 

import pandas as pd
import numpy as np

# 1. Cargar dataset original de Chicago Crimes
df = pd.read_csv("Chicago_Crimes_2012_to_2017.csv", low_memory=False)
print("Shape inicial:", df.shape)  # Muestra tamaño original

# 2. Selección de columnas relevantes para la práctica
cols = [
    "ID", "Case Number", "Date", "Primary Type", "Description",
    "Location Description", "Arrest", "Domestic",
    "District", "Ward", "Community Area", "Latitude", "Longitude"
]
df = df[cols]

# 3. Renombrar columnas a nombres más legibles y al español
df = df.rename(columns={
    "ID": "ID",
    "Case Number": "Número de Caso",
    "Date": "Fecha",
    "Primary Type": "Tipo Principal",
    "Description": "Descripción",
    "Location Description": "Ubicación",
    "Arrest": "¿Hubo Arresto?",
    "Domestic": "Violencia Doméstica",
    "District": "Distrito",
    "Ward": "Barrio",
    "Community Area": "Área Comunitaria",
    "Latitude": "Latitud",
    "Longitude": "Longitud"
})

# 4. Convertir Arresto y Violencia Doméstica a valores "Sí"/"No"
df["¿Hubo Arresto?"] = df["¿Hubo Arresto?"].apply(lambda x: "Sí" if x else "No")
df["Violencia Doméstica"] = df["Violencia Doméstica"].apply(lambda x: "Sí" if x else "No")

# 5. Limpieza y descomposición de fechas
df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")  # Convertir a datetime
df["Año"] = df["Fecha"].dt.year
df["Mes"] = df["Fecha"].dt.month
df["Día"] = df["Fecha"].dt.day
df["Hora"] = df["Fecha"].dt.hour

# Obtener día de la semana en inglés minúscula
df["Día de la Semana"] = df["Fecha"].dt.day_name().str.lower()

# Crear columna ¿Es Fin de Semana? (True/False inicialmente)
df["¿Es Fin de Semana?"] = df["Día de la Semana"].isin(["saturday", "sunday"])

# Traducir día de la semana a español
dias = {
    "monday": "Lunes", "tuesday": "Martes", "wednesday": "Miércoles",
    "thursday": "Jueves", "friday": "Viernes", "saturday": "Sábado", "sunday": "Domingo"
}
df["Día de la Semana"] = df["Día de la Semana"].map(dias)

# Convertir columna fin de semana a "Sí"/"No"
df["¿Es Fin de Semana?"] = df["¿Es Fin de Semana?"].apply(lambda x: "Sí" if x else "No")

# 6. Asegurar que las columnas numéricas tengan el tipo correcto
numeric_cols = ["Distrito", "Barrio", "Área Comunitaria", "Latitud", "Longitud"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 7. Eliminar duplicados y filas con valores nulos
df = df.drop_duplicates(subset="ID")
df = df.dropna()

# 8. Reducir tamaño del dataset a 7000 filas para práctica
df = df.sample(n=7000, random_state=42)
print("Shape final (muestra):", df.shape)

# 9. Guardar dataset limpio y reducido en CSV
df.to_csv("Crimenes_Chicago_Limpio_7000.csv", index=False)
print("Dataset reducido guardado como Crímenes_Chicago_Limpio_7000.csv")

