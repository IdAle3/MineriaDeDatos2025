import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import os


def generate_wordcloud(df, column, output_path):
    """Genera un Word Cloud a partir de una columna del DataFrame y guarda la imagen."""
    
    texto = " ".join(df[column].dropna().astype(str).tolist()).lower()

    # Mostrar top 10 palabras más comunes, aquí se quitan las palabras STOPWORDS para evitar que salga algo 
    # como "the", "to" en las palabras más comunes
    palabras = texto.split()
    print("\n Top 10 palabras más frecuentes:")
    palabras_filtradas = [p for p in palabras if p not in STOPWORDS]
    print(Counter(palabras_filtradas).most_common(10))

    # Generar Word Cloud

    wordcloud = WordCloud(
        background_color="white",
        width=800,
        height=400,
        min_font_size=6,
        stopwords=STOPWORDS
    ).generate(texto)

    # Graficar y guardar la imagen
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"\n Word Cloud generado en: {output_path}")


def main():
    # Rutas de tu proyecto
    csv_path = "Practica1/Crimenes_Chicago_Limpio_7000.csv"
    output_folder = "Practica9/img"
    output_image = f"{output_folder}/word_cloud_descripcion.png"

    # Crear carpeta si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Cargar dataset
    df = pd.read_csv(csv_path)

    # Llamar a la función del wordcloud
    generate_wordcloud(df, "Descripción", output_image)


# Ejecutar script
if __name__ == "__main__":
    main()
