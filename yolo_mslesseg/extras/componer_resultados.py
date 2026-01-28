"""
Script: componer_resultados.py

Descripción:

    Genera una tabla resumen a partir de los resultados globales obtenidos
    en distintos experimentos. Este script busca automáticamente los
    archivos JSON generados por `promediar_folds.py` y compone un único
    CSV con todas las métricas globales agrupadas por plano anatómico y
    algoritmo de mejora.

    Este script requiere especificar una combinación exacta de parámetros
    del experimento (modalidad, número de cortes, k_folds y número de épocas
    de entrenamiento). Estos parámetros definen qué subdirectorio de results/
    se debe procesar. Los campos `plano` y `mejora` se incluyen únicamente
    para construir el objeto `Modelo`, pero no influyen en el filtrado de
    resultados, ya que el CSV final siempre incorpora todos los planos y
    todas las mejoras disponibles para esa configuración.

Modo de ejecución:
    Este script debe ejecutarse únicamente por CLI. No es parte del pipeline, por lo que
    no está preparado para uso interno.

Argumentos CLI:
    --plano (str, requerido)
        Plano anatómico del modelo ('axial', 'coronal', 'sagital').
        Se usa solo para construir el objeto Modelo.
        No filtra los planos incluidos en el CSV final.

    --modalidad (list[str], opcional)
        Modalidad o modalidades de imagen MRI ('T1', 'T2', 'FLAIR').
        Por defecto todas. Se usa para localizar el experimento.

    --num_cortes (int_o_percentil, requerido)
        Número de cortes extraídos (valor entero o percentil, por ejemplo 50 o 'P50').

    --mejora (str, opcional)
        Algoritmo de mejora aplicado ('HE', 'CLAHE', 'GC', 'LT', o None).
        Por defecto None. No afecta al filtrado, solo se usa para localizar el experimento.

    --epochs (int, requerido)
        Número de épocas del modelo entrenado.

    --k_folds (int, opcional)
        Número de folds usados en validación cruzada.
        Por defecto 5.

    --limpiar (flag, opcional)
        Si existe, elimina el archivo de salida existente antes de crear uno nuevo.

Uso por CLI:
    python -m yolo_mslesseg.extras.componer_resultados \
        --plano axial \
        --modalidad FLAIR \
        --num_cortes P50 \
        --epochs 50 \
        --k_folds 5

Entradas:
    - Archivos JSON con métricas globales generados por `promediar_folds.py` para un
      experimento concreto:
            results/<mejora>/<modalidad>_<num_cortes>c_<k_folds>folds_<epochs>epochs/global<plano>_results.json

    - Clases:
        * Modelo → se instancia solo para recuperar `model_string` y localizar la carpeta correcta.

Salidas:
    - CSV resumen con las columnas
        Plano | Mejora | DSC (mean ± std) | AUC (mean ± std) |
        Precision (mean ± std) | Recall (mean ± std)
      en
      <modalidad>_<num_cortes>c_<k_folds>folds_<epochs>epochs_results.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from yolo_mslesseg.utils.Modelo import Modelo
from yolo_mslesseg.utils.configurar_logging import get_logger
from yolo_mslesseg.utils.utils import (
    leer_json,
    ruta_existente,
    int_o_percentil,
    construir_nombre_configuracion,
)

logger = get_logger(__file__)

# =============================
#       FUNCIONES BASE
# =============================


def listar_jsons(directorio):
    """Devuelve lista de archivos global_*.json."""
    if not directorio.exists():
        return []
    return [
        f
        for f in directorio.iterdir()
        if f.is_file() and f.suffix == ".json" and f.name.startswith("global_")
    ]


def parsear_experimento(filepath):
    """Obtiene (plano, mejora)."""
    partes = filepath.stem.split("_")
    plano = next(
        (
            p.capitalize()
            for p in partes
            if p.lower() in ["axial", "coronal", "sagital", "consenso"]
        ),
        "Desconocido",
    )

    for parent in filepath.parents:
        nombre = parent.name.upper()
        if nombre in ["CONTROL", "HE", "CLAHE", "GC", "LT"]:
            return plano, nombre

    raise ValueError(f"No se pudo inferir la mejora desde la ruta: {filepath}")


def formatear_valor_json(valor):
    """Convierte dict con media/std → 'media ± std'."""
    if isinstance(valor, dict) and "media" in valor and "std" in valor:
        return f"{float(valor['media']):.3f} ± {float(valor['std']):.3f}"
    return ""


def leer_metricas_json(filepath):
    """Lee JSON y devuelve {metrica: 'X ± Y'}."""
    metricas = {}

    try:
        data = leer_json(filepath)
    except Exception as e:
        logger.warning(f"⚠️ No se pudo leer el archivo JSON {filepath}: {e}.")
        return metricas

    for metrica, valores in data.items():
        metricas[metrica] = formatear_valor_json(valores)

    return metricas


def construir_fila(plano, mejora, metricas):
    """Arma fila para el DataFrame."""
    return {
        "Mejora": mejora,
        "Plano": plano,
        "DSC (mean ± std)": metricas.get("DSC", ""),
        "AUC (mean ± std)": metricas.get("AUC", ""),
        "Precision (mean ± std)": metricas.get("Precision", ""),
        "Recall (mean ± std)": metricas.get("Recall", ""),
    }


def ordenar_dataframe(df):
    """Ordena por plano y mejora."""
    orden_plano = ["Axial", "Coronal", "Sagital", "Consenso"]
    orden_mejora = ["Base", "HE", "CLAHE", "GC", "LT"]

    df["Plano"] = pd.Categorical(df["Plano"], categories=orden_plano, ordered=True)
    df["Mejora"] = pd.Categorical(df["Mejora"], categories=orden_mejora, ordered=True)
    df.sort_values(by=["Mejora", "Plano"], inplace=True)


# =============================
#        PROCESAMIENTO
# =============================


def componer_resultados(configuracion_global):
    """Busca los JSON del experimento solicitado y crea el CSV final."""

    filas = []
    lista_jsons = []

    logger.info(
        f"🔍 Buscando experimentos dentro de la configuración global {configuracion_global}"
    )

    results_dir = Path("results")
    output_path = results_dir / f"{configuracion_global}_results.csv"

    # Recorrer todas las mejoras
    for mejora_dir in results_dir.iterdir():
        if not mejora_dir.is_dir():
            continue

        experimento_dir = mejora_dir / configuracion_global
        if experimento_dir.exists():
            lista_jsons.extend(listar_jsons(experimento_dir))

    if not lista_jsons:
        logger.warning(f"⚠️ No se encontraron JSONs para: {configuracion_global}")
        return None

    logger.info(f"📂 Se encontraron {len(lista_jsons)} archivos JSON")

    # Procesar cada JSON
    for json_file in lista_jsons:
        plano, mejora = parsear_experimento(json_file)
        metricas = leer_metricas_json(json_file)
        filas.append(construir_fila(plano, mejora, metricas))

    if not filas:
        logger.warning("⚠️ No se generaron filas válidas.")
        return None

    df = pd.DataFrame(filas)
    ordenar_dataframe(df)
    df["Mejora"] = df["Mejora"].replace("CONTROL", "Base").fillna("Base")

    df.to_csv(output_path, index=False)
    logger.info(f"✅ Resumen exportado en {output_path}")

    return df


# =============================
#        FLUJO PRINCIPAL
# =============================


def ejecutar_flujo(modelo, epochs, limpiar):
    """
    Ejecuta el flujo de composición de resultados.
    """
    logger.header(f"📊️ Generando tabla de resultados")

    configuracion_global = construir_nombre_configuracion(modelo, epochs)

    output_path = Path("results") / f"{configuracion_global}_results.csv"

    # Limpiar si corresponde
    if limpiar and ruta_existente(output_path):
        logger.info(f"♻️ Eliminando tabla previa.")
        output_path.unlink(missing_ok=True)

    componer_resultados(configuracion_global)


# =============================
#       CLI Y EJECUCIÓN
# =============================


def parsear_args():
    """
    Parsea los argumentos del script
    leyendolos desde la línea de comandos (CLI).
    """
    parser = argparse.ArgumentParser(
        description="Componer los resultados globales del experimento seleccionado."
    )
    parser.add_argument(
        "--plano",
        type=str,
        required=True,
        choices=["axial", "coronal", "sagital"],
        metavar="[axial, coronal, sagital]",
        help="Plano anatómico de extracción.",
    )
    parser.add_argument(
        "--modalidad",
        nargs="+",
        choices=["T1", "T2", "FLAIR"],
        default=["T1", "T2", "FLAIR"],
        metavar="[T1, T2, FLAIR]",
        help="Modalidad(es) de imagen MRI. Por defecto todas.",
    )
    parser.add_argument(
        "--num_cortes",
        type=int_o_percentil,
        required=True,
        metavar="<num_cortes>",
        help="Número de cortes extraídos (valor fijo o percentil).",
    )
    parser.add_argument(
        "--mejora",
        type=str,
        default=None,
        choices=["HE", "CLAHE", "GC", "LT"],
        metavar="<mejora>",
        help="Algoritmo de mejora de imagen aplicado. Por defecto None.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        metavar="<epochs>",
        help="Número de épocas de entrenamiento.",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        metavar="<k_folds>",
        help="Número de folds para validación cruzada. Por defecto 5.",
    )
    parser.add_argument(
        "--limpiar",
        action="store_true",
        default=False,
        help="Limpiar el resultado generado previamente.",
    )

    return parser.parse_args()


def main():
    """
    Entrada CLI del script: parsea argumentos, construye Modelo
    y ejecuta el flujo completo.
    """
    args = parsear_args()

    modelo = Modelo(
        plano=args.plano,
        num_cortes=args.num_cortes,
        modalidad=args.modalidad,
        k_folds=args.k_folds,
        mejora=args.mejora,
    )

    ejecutar_flujo(
        modelo=modelo,
        epochs=args.epochs,
        limpiar=args.limpiar,
    )


if __name__ == "__main__":
    main()
