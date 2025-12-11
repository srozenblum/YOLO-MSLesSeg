"""
Script: promediar_folds.py

Descripción:
    Calcula el promedio y la desviación estándar de las métricas obtenidas
    en cada fold de validación cruzada, generando un archivo JSON con los
    resultados globales del experimento.  Puede ejecutarse tanto de forma
    independiente (desde CLI) como internamente dentro del pipeline
    (`ejecutar_pipeline.py`).

    Este script debe ejecutarse tras la evaluación de todos los folds
    para resumir el rendimiento global de un modelo.

Modos de ejecución:
    1. CLI (uso independiente):
       - Se leen y parsean los argumentos escritos por el usuario en la línea de comandos.
       - Se crea la instancia de Modelo.

    2. Interno (desde `ejecutar_pipeline.py`):
       - Se recibe la instancia ya creada de Modelo.
       - No se usa el parser de argumentos.

Argumentos CLI:
    --plano (str, requerido)
        Plano anatómico de extracción ('axial', 'coronal', 'sagital').

    --modalidad (list[str], opcional)
        Modalidad o modalidades de imagen MRI ('T1', 'T2', 'FLAIR').
        Por defecto todas.

    --num_cortes (int_o_percentil, requerido)
        Número de cortes extraídos (valor entero o percentil, por ejemplo 50 o 'P75').

    --mejora (str, opcional)
        Algoritmo de mejora de imagen aplicado ('HE', 'CLAHE', 'GC', 'LT', o None).
        Por defecto None.

    --epochs (int, requerido)
        Número de épocas del modelo entrenado.

    --k_folds (int, opcional)
        Número de folds para validación cruzada.
        Por defecto 5.

    --limpiar (flag, opcional)
        Limpiar los resultados globales previos antes de calcular nuevos.

Uso por CLI:
        python -m yolo_mslesseg.scripts.promediar_folds \
        --plano "coronal" \
        --num_cortes 40 \
        --epochs 80 \
        --k_folds 5

Entradas:
    - Archivos JSON con métricas por fold: generados previamente por `eval.py` en
        results/<modalidad>_<num_cortes>c_<k_folds>folds_<epochs>epochs/foldX/foldX_<plano>_results.json.

Salidas:
    - JSON con métricas globales del experimento (promedio y desviación estándar) en
        results/<modalidad>_<num_cortes>c_<k_folds>folds_<epochs>epochs/<plano>_global_results.json
"""

import argparse
import sys

import numpy as np

from yolo_mslesseg.configs.ConfigEval import ConfigEval
from yolo_mslesseg.utils.Modelo import Modelo
from yolo_mslesseg.utils.utils import (
    int_o_percentil,
    ruta_existente,
    escribir_json,
    leer_json,
    get_logger,
)

# Configurar logger
logger = get_logger(__file__)

# =============================
#       FUNCIONES BASE
# =============================


def agregar_metricas_fold(dic_total, archivo):
    """Agrega las métricas leídas de un fold al diccionario acumulador."""
    metricas = leer_json(archivo)
    for k, v in metricas.items():
        # Caso 1: formato de fold {"media": x, "std": y}
        if isinstance(v, dict) and "media" in v:
            dic_total.setdefault(k, []).append(v["media"])
        # Caso 2: formato de paciente {"metrica": valor}
        elif isinstance(v, (int, float)):
            dic_total.setdefault(k, []).append(float(v))
        else:
            logger.warning(
                f"⚠️ Formato inesperado para la métrica '{k}' en {archivo}: {v}"
            )


def leer_metricas_folds(config):
    """
    Lee los archivos de métricas de cada fold y acumula sus valores en un diccionario.
    Devuelve un diccionario {métrica: [valores_por_fold]}.
    """
    metricas_fold = {}

    for fold_dir in config.results_base_dir.iterdir():
        if fold_dir.is_dir():
            results_json = fold_dir / f"{fold_dir.name}_{config.plano}_results.json"

            if not ruta_existente(results_json):
                logger.warning(f"⚠️ No se encontró {results_json}, se omite este fold.")
                continue

            agregar_metricas_fold(dic_total=metricas_fold, archivo=results_json)

    if not metricas_fold:
        logger.warning("⚠️ No se encontraron métricas válidas en los folds.")

    return metricas_fold


def calcular_resumen_experimento(metricas_fold):
    """Calcula la media y desviación estándar para cada métrica a partir de los folds."""
    resultados = {}
    for metrica, valores in metricas_fold.items():
        resultados[metrica] = {
            "media": float(np.round(np.mean(valores), 3)),
            "std": float(np.round(np.std(valores, ddof=1), 3)),
        }
    return resultados


def exportar_resultados_experimento(resultados, output_path):
    """Guarda el resumen global del experimento en formato JSON."""
    if not resultados:
        logger.warning("⚠️ No hay resultados para exportar.")
        return
    escribir_json(dic=resultados, json_path=output_path)


# =============================
#        PROCESAMIENTO
# =============================


def procesar_resultados(config):
    """Calcula las métricas promedio del experimento a partir de los resultados de cada fold."""
    # Ruta del archivo de resultados del experimento
    output_path = config.results_base_dir / f"global_{config.plano}_results.json"

    # Evitar recalcular si ya existe
    if ruta_existente(output_path):
        return

    # Leer resultados de cada fold y calcular el promedio
    metricas_fold = leer_metricas_folds(config)
    resultados_experimento = calcular_resumen_experimento(metricas_fold)

    exportar_resultados_experimento(resultados_experimento, output_path)
    return resultados_experimento


# =============================
#        FLUJO PRINCIPAL
# =============================


def ejecutar_flujo_promediar(config, limpiar, verbose=False):
    """
    Ejecuta el flujo principal de cálculo de métricas promedio de folds.
    """
    if verbose:
        logger.header(f"🧮 Promediando folds para el fold {config.fold_test}.")

    if limpiar:
        if verbose:
            logger.info(f"♻️ Limpiando promedio de métricas previo.")
        config.limpiar_resultados()

    config.verificar_paths()

    metricas_experimento = procesar_resultados(config=config)

    if metricas_experimento is None:  # Ningún resultado procesado
        logger.skip(f"⏩ Promedio de folds ya existente.")
    elif len(metricas_experimento) > 0:  # Todos los resultados procesados
        logger.info(f"🆗 Promedio de folds calculado correctamente.")
    else:
        logger.warning("⚠️ Estado desconocido al calcular promedio de folds.")


# =============================
#       CLI Y EJECUCIÓN
# =============================


def parsear_args(argv=None):
    """
    Parsea los argumentos del script.
    Si no se pasa una lista de argumentos, se leen los utilizados
    al ejecutar el script desde la terminal.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Calcular los resultados globales del experimento a partir del promedio de los resultados de cada fold.",
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
        metavar="",
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
        help="Limpiar los resultados generados previamente.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    """
    Entrada CLI del script: parsea argumentos, construye Modelo/Paciente/ConfigEval
    y ejecuta el flujo completo.
    """
    args = parsear_args(argv)

    modelo = Modelo(
        plano=args.plano,
        num_cortes=args.num_cortes,
        modalidad=args.modalidad,
        k_folds=args.k_folds,
        mejora=args.mejora,
    )

    config = ConfigEval(
        modelo=modelo,
        epochs=args.epochs,
        k_folds=args.k_folds,
        fold_test=args.fold_test,
    )

    ejecutar_flujo_promediar(config=config, limpiar=args.limpiar, verbose=True)


def ejecutar_promediar_folds_pipeline(
    modelo, plano=None, epochs=50, k_folds=5, limpiar=False
):
    """
    Entrada interna para el pipeline: recibe objetos ya
    construidos y ejecuta el flujo sin usar el parser CLI.
    """
    config = ConfigEval(
        modelo=modelo,
        epochs=epochs,
        k_folds=k_folds,
        fold_test=None,
        plano_forzado=plano,
    )

    ejecutar_flujo_promediar(
        config=config,
        limpiar=limpiar,
    )


if __name__ == "__main__":
    main()
