# Práctica 4 - Statistical Test
# ANOVA, T-test y Kruskal-Wallis en crímenes de Chicago

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from scipy.stats import shapiro, norm
import statsmodels.api as sm

# === Configuración ===
OUTPUT_DIR = "img"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# === Funciones ===
def load_data(path="../Practica1/Crimenes_Chicago_Limpio_7000.csv"):
    """Carga y prepara el dataset."""
    df = pd.read_csv(path)
    df["Distrito"] = df["Distrito"].astype(int)
    df["Año"] = df["Año"].astype(int)
    return df


def normality_test(data, alpha=0.05):
    """Aplica Shapiro-Wilk a una serie de datos."""
    stat, p = shapiro(data)
    return p > alpha, stat, p


def test_normality_by_district(df, distritos, alpha=0.05):
    """Prueba de normalidad para cada distrito."""
    results = {}
    print("\n=== Pruebas de Normalidad (Shapiro-Wilk) ===")
    for d in distritos:
        data = df[df["Distrito"] == d]["Año"]
        is_normal, stat, p = normality_test(data, alpha)
        results[d] = is_normal
        print(f"Distrito {d}: Estadístico={stat:.4f}, p-valor={p:.3e}, Normal={is_normal}")
    return results


def compare_groups(df, distritos, normal_results):
    """Decide entre ANOVA y Kruskal-Wallis."""
    grupos = [df[df["Distrito"] == d]["Año"] for d in distritos]
    if all(normal_results.values()):
        res = stats.f_oneway(*grupos)
        print("\n=== ANOVA ===")
        print("Estadístico F:", res.statistic)
        print("p-valor:", res.pvalue)
    else:
        res = stats.kruskal(*grupos)
        print("\n=== Kruskal-Wallis ===")
        print("Estadístico H:", res.statistic)
        print("p-valor:", res.pvalue)
        alpha = 0.05  # Nivel de significancia
    if res.pvalue < alpha:
        print(f"Conclusión: p < {alpha}. Rechazamos la hipótesis nula → al menos un grupo presenta diferencias significativas respecto a los demás. "
            f"Esto indica que los grupos no tienen distribuciones iguales y al menos uno se comporta de manera distinta.")
    else:
        print(f"Conclusión: p >= {alpha}. No se rechaza la hipótesis nula → no hay evidencia suficiente para afirmar que los grupos difieren. "
            f"Esto sugiere que las distribuciones de los grupos son similares y no se observan diferencias significativas.")



def pairwise_test(df, d1, d2, alpha=0.05):
    """Compara los dos distritos principales con T-test o Mann-Whitney."""
    data1 = df[df["Distrito"] == d1]["Año"]
    data2 = df[df["Distrito"] == d2]["Año"]
    d1_normal, _, _ = normality_test(data1, alpha)
    d2_normal, _, _ = normality_test(data2, alpha)
    alpha = 0.05  # Nivel de significancia

    if d1_normal and d2_normal:
        # T-Test de Welch
        res = stats.ttest_ind(data1, data2, equal_var=False)
        test_name = "T-Test (Welch)"
        stat_name = "t"
    else:
        # Mann-Whitney U
        res = stats.mannwhitneyu(data1, data2)
        test_name = "Mann-Whitney U"
        stat_name = "U"

    # Resultados
    print(f"\n=== {test_name} ===")
    print(f"Distrito {d1} normal: {d1_normal}")
    print(f"Distrito {d2} normal: {d2_normal}")
    print(f"Distrito {d1} vs {d2} -> Estadístico {stat_name} = {res.statistic:.4f}, p-valor = {res.pvalue:.4e}")

    # Conclusión 
    if res.pvalue < alpha:
        if test_name == "T-Test (Welch)":
            print(f"\n=== {test_name} ===\nDistrito {d1} normal: {d1_normal}\nDistrito {d2} normal: {d2_normal}\n"
                f"Distrito {d1} vs {d2} -> Estadístico {stat_name} = {res.statistic:.4f}, p-valor = {res.pvalue:.4e}\n"
                f"Conclusión: La diferencia en las medias de {d1} y {d2} es estadísticamente significativa "
                f"(p < {alpha}), lo que indica que los dos distritos tienen un comportamiento distinto en la variable analizada.")
        else:
            print(f"\n=== {test_name} ===\nDistrito {d1} normal: {d1_normal}\nDistrito {d2} normal: {d2_normal}\n"
                f"Distrito {d1} vs {d2} -> Estadístico {stat_name} = {res.statistic:.4f}, p-valor = {res.pvalue:.4e}\n"
                f"Conclusión: La diferencia en la distribución/mediana de {d1} y {d2} es estadísticamente significativa "
                f"(p < {alpha}), lo que indica que los dos distritos presentan patrones distintos en la variable analizada.")
    else:
        print(f"\n=== {test_name} ===\nDistrito {d1} normal: {d1_normal}\nDistrito {d2} normal: {d2_normal}\n"
            f"Distrito {d1} vs {d2} -> Estadístico {stat_name} = {res.statistic:.4f}, p-valor = {res.pvalue:.4e}\n"
            f"Conclusión: No se encontró evidencia suficiente de diferencias entre {d1} y {d2} "
            f"(p ≥ {alpha}), lo que sugiere que ambos distritos tienen un comportamiento similar en la variable analizada.")



def plot_histograms(df, distritos, normal_results, output_dir):
    """Genera histogramas individuales por distrito."""
    for d in distritos:
        plt.figure(figsize=(10, 6))
        subset = df[df["Distrito"] == d]["Año"]

        sns.histplot(subset, bins=15, kde=False, alpha=0.7,
                     edgecolor='black', stat='density', color='purple', zorder=1)
        sns.kdeplot(subset, color="purple", linewidth=2, zorder=2, label="KDE")

        if normal_results[d]:
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, subset.mean(), subset.std())
            plt.plot(x, p, 'k', linewidth=2, label='Distribución normal', zorder=3)

        plt.title(f"Distribución de años - Distrito {d}\n(Shapiro-Wilk p={shapiro(subset)[1]:.3e})")
        plt.xlabel("Año")
        plt.ylabel("Densidad")
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
        plt.legend()
        plt.savefig(f"{output_dir}/histograma_kde_distrito_{d}.png", dpi=300, bbox_inches='tight')
        plt.close()


def plot_kde_comparison(df, distritos, output_dir):
    """Genera una gráfica KDE combinada de todos los distritos."""
    plt.figure(figsize=(12, 8))
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i, d in enumerate(distritos):
        subset = df[df["Distrito"] == d]["Año"]
        sns.kdeplot(subset, label=f"Distrito {d}", linewidth=2.5, color=colors[i])

    plt.title("Comparación de distribuciones de años por distrito (KDE)")
    plt.xlabel("Año")
    plt.ylabel("Densidad")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
    plt.savefig(f"{output_dir}/kde_comparacion_distritos.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_matrix(df, distritos, output_dir):
    """Matriz de gráficos con histogramas y KDE por distrito + comparación final."""
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    for i, d in enumerate(distritos):
        subset = df[df["Distrito"] == d]["Año"]
        sns.histplot(subset, bins=15, kde=True, ax=axes[i],
                     alpha=0.7, edgecolor='black', stat='density', color=colors[i])
        axes[i].set_title(f"Distrito {d}")
        axes[i].set_xlabel("Año")
        axes[i].set_ylabel("Densidad")
        axes[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))

    for d in distritos:
        subset = df[df["Distrito"] == d]["Año"]
        sns.kdeplot(subset, ax=axes[5], label=f"Distrito {d}", linewidth=2)

    axes[5].set_title("Todas las distribuciones (KDE)")
    axes[5].set_xlabel("Año")
    axes[5].set_ylabel("Densidad")
    axes[5].legend()
    axes[5].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/matriz_comparacion_distritos.png", dpi=300, bbox_inches='tight')
    plt.close()


# === Main ===
if __name__ == "__main__":
    df = load_data()
    top_distritos = df["Distrito"].value_counts().head(5).index
    df_top = df[df["Distrito"].isin(top_distritos)]
    print("Distritos analizados:", list(top_distritos))

    normal_results = test_normality_by_district(df_top, top_distritos)
    compare_groups(df_top, top_distritos, normal_results)
    pairwise_test(df_top, top_distritos[0], top_distritos[1])

    plot_histograms(df_top, top_distritos, normal_results, OUTPUT_DIR)
    plot_kde_comparison(df_top, top_distritos, OUTPUT_DIR)
    plot_matrix(df_top, top_distritos, OUTPUT_DIR)

    print(f"\nResultados guardados en carpeta: {OUTPUT_DIR}")
