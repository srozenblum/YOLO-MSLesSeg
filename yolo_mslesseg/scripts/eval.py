"""
Script: eval.py

Descripción:
    Evalúa el rendimiento de un modelo YOLO, ya sea para un paciente
    individual o para todos los pacientes de un fold. Puede ejecutarse
    tanto de forma independiente (desde CLI) como internamente dentro
    del pipeline (`ejecutar_pipeline.py`).

    Calcula métricas de segmentación (DSC, AUC, precisión, recall) a
    partir de los volúmenes reconstruidos y genera archivos JSON con
    los resultados.

Modos de ejecución:
    1. CLI (uso independiente):
       - Se leen y parsean los argumentos escritos por el usuario en la línea de comandos.
       - Se crean las instancias de Modelo y, opcionalmente, Paciente.

    2. Interno (desde `ejecutar_pipeline.py`):
       - Se reciben instancias ya creadas de Modelo y (opcionalmente) Paciente,
       junto con el resto de parámetros.
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

    --fold_test (int, excluyente con --paciente_id)
        Calcular las métricas para todos los pacientes del
        fold indicado, correspondiente al conjunto de test.

    --paciente_id (str, excluyente con --fold_test)
        Calcular las métricas solo para el paciente indicado.

    --limpiar (flag, opcional)
        Limpiar el directorio con las predicciones 2D binarias antes de generar nuevas.

Uso por CLI:
    python -m yolo_mslesseg.scripts.eval \
        --plano "axial" \
        --modalidad "FLAIR" \
        --num_cortes P50 \
        --epochs 60 \
        --fold_test 5 \
        --limpiar

Entradas:
    - Volúmenes predichos (.nii.gz): generados previamente por `reconstruir_volumen.py` en
        pred_vols/<mejora>/<modalidad>_<num_cortes>c_<k_folds>folds_<epochs>epochs/<fold_test>/PX/.

    - Ground truth (.nii.gz): volúmenes originales ubicados en GT/<paciente_id>/
        utilizados como referencia para la evaluación.

    - Clases:
        * ConfigEval → gestiona rutas y variables globales asociadas a la evaluación.
        * Modelo → define el plano, modalidades, mejora y número de cortes del experimento.

Salidas:
    - JSON con métricas por paciente o promedio del fold en
        results/<modalidad>_<num_cortes>c_<k_folds>folds_<epochs>epochs/foldX/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from yolo_mslesseg.configs.ConfigEval import ConfigEval
from yolo_mslesseg.utils.Modelo import Modelo
from yolo_mslesseg.utils.Paciente import Paciente
from yolo_mslesseg.utils.configurar_logging import get_logger
from yolo_mslesseg.utils.utils import (
    int_o_percentil,
    cargar_volumen,
    ruta_existente,
    listar_pacientes,
    reconstruccion_valida,
    escribir_json,
    log_estado_fold,
    leer_json,
    AUC,
    precision,
    recall,
    DSC,
)

# Configurar logger
logger = get_logger(__file__)


# =============================
#     FUNCIONES AUXILIARES
# =============================


def generar_diccionario_metricas(gt_vol, pred_vol):
    """
    Calcula un diccionario de métricas (DSC, AUC, Precision, Recall)
    a partir del volumen predicho y el volumen ground truth.
    """

    metricas = {
        "DSC": DSC(gt_vol, pred_vol),
        "AUC": AUC(gt_vol, pred_vol),
        "Precision": precision(gt_vol, pred_vol),
        "Recall": recall(gt_vol, pred_vol),
    }

    return metricas


def calcular_metricas(gt_vol_path, pred_vol_path):
    """Carga los volúmenes, los valida y calcula las métricas."""
    # Validar la reconstrucción antes de cargar
    if not reconstruccion_valida(pred_vol_path, gt_vol_path):
        logger.warning(f"⚠️ Reconstrucción inválida: {Path(pred_vol_path).name}")
        return {}

    pred_vol = cargar_volumen(pred_vol_path)
    gt_vol = cargar_volumen(gt_vol_path)

    return generar_diccionario_metricas(gt_vol, pred_vol)


def calcular_promedio(metricas_dic):
    """
    Calcula el promedio y la desviación estándar para
    todas las métricas en un diccionario.
    """
    if not metricas_dic:
        raise ValueError("El diccionario de métricas está vacío.")

    metricas_promedio = {
        metrica: {
            "media": float(np.round(np.mean(valor), 3)),
            "std": float(np.round(np.std(valor), 3)),
        }
        for metrica, valor in metricas_dic.items()
    }

    return metricas_promedio


# =============================
#        PROCESAMIENTO
# =============================


def procesar_paciente_eval(config, paths_dir=None, modo_fold=False):
    """Ejecuta el cálculo de métricas para un paciente individual."""
    # Si no se pasan los directorios → modo paciente → asigna los del config
    if paths_dir is None:
        paths_dir = {
            "pred_vol": config.paciente_pred_vol,
            "gt_vol": config.paciente_gt_vol,
            "results_json": config.paciente_results_json,
        }

    pred_vol_paciente = paths_dir["pred_vol"]
    gt_vol_paciente = paths_dir["gt_vol"]
    results_json_paciente = paths_dir["results_json"]

    # Si ya existe el JSON de métricas
    if ruta_existente(results_json_paciente):
        if modo_fold:
            return leer_json(results_json_paciente)
        return None  # Llamada directa → no recalcular

    # Si no existe, calcular métricas nuevas
    metricas_dic = calcular_metricas(
        gt_vol_path=gt_vol_paciente, pred_vol_path=pred_vol_paciente
    )
    escribir_json(dic=metricas_dic, json_path=results_json_paciente)

    return metricas_dic


def construir_paths(paciente_id, config):
    """
    Construye un diccionario de paths (pred_vol, gt_vol, metricas_json)
    para un paciente individual.
    """
    root_pred_vols = config.pred_vols_fold_dir / paciente_id
    root_gt = config.gt_dir / paciente_id
    root_res = config.results_fold_dir / paciente_id

    return {
        "pred_vol": root_pred_vols / f"{paciente_id}_{config.plano}.nii.gz",
        "gt_vol": root_gt / f"{paciente_id}_MASK.nii.gz",
        "results_json": root_res / f"{paciente_id}_{config.plano}_results.json",
    }


def calcular_metricas_fold(input_dir, config):
    """
    Calcula las métricas por paciente y el promedio del fold.
    """
    output_path = config.results_fold_json

    # Evitar recalcular si ya existen resultados
    if ruta_existente(output_path):
        return

    pacientes = listar_pacientes(input_dir)
    metricas_fold = {}

    for paciente_id in pacientes:
        paths_paciente = construir_paths(paciente_id, config)
        metricas_paciente = procesar_paciente_eval(
            config=config, paths_dir=paths_paciente, modo_fold=True
        )
        if not metricas_paciente:
            logger.warning(f"⚠️ No se encontraron métricas del paciente {paciente_id}.")
            continue

        # Acumular métricas para el fold
        for metrica, valor in metricas_paciente.items():
            metricas_fold.setdefault(metrica, []).append(valor)

    # Calcular media y std y guardarlas en JSON
    metricas_stats = calcular_promedio(metricas_fold)
    escribir_json(dic=metricas_stats, json_path=output_path)

    return metricas_stats


# =============================
#        FLUJO PRINCIPAL
# =============================


def ejecutar_flujo_eval(config, limpiar, verbose=False):
    """
    Ejecuta el flujo principal de evaluación,
    ya sea sobre un fold completo o un paciente individual.
    """
    if verbose:
        str_fold = f"fold {config.fold_test}"
        str_paciente = f"paciente {config.paciente}"
        logger.header(
            f"\n📈 Calculando métricas ({config.plano}) para el {str_paciente if config.es_paciente_individual else str_fold}."
        )

    # Limpiar si corresponde
    if limpiar:
        if verbose:
            logger.info(f"♻️ Limpiando resultados previos.")
        config.limpiar_resultados()

    # Verificar paths
    config.verificar_paths()

    # Ejecucion por paciente
    if config.es_paciente_individual:
        metricas_paciente = procesar_paciente_eval(config=config)
        if metricas_paciente is None:
            logger.skip(f"⏩ Métricas ya existentes.")
        elif isinstance(metricas_paciente, (dict, list)):
            logger.info(f"✅ Métricas calculadas correctamente.")
        else:
            logger.warning(f"⚠️ Estado desconocido al calcular métricas.")

    # Ejecucion por fold
    else:
        metricas_fold = calcular_metricas_fold(
            input_dir=config.pred_vols_fold_dir, config=config
        )
        log_estado_fold(logger=logger, resultado=metricas_fold, fold=config.fold_test)


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
        description="Evaluar un modelo entrenado sobre el conjunto de test usando DSC, AUC, precision y recall.",
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--fold_test",
        type=int,
        metavar="<fold_test>",
        help="Calcular las métricas para el fold indicado, utilizado como conjunto de test.",
    )
    group.add_argument(
        "--paciente_id",
        type=str,
        metavar="<paciente_id>",
        help="Calcular las métricas solo para el paciente indicado.",
    )
    parser.add_argument(
        "--limpiar",
        action="store_true",
        default=False,
        help="Limpiar los resultados calculados previamente.",
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

    # Ejecución por paciente
    if args.paciente_id is not None:
        paciente = Paciente(
            id=args.paciente_id,
            plano=modelo.plano,
            modalidad=modelo.modalidad,
            mejora=modelo.mejora,
        )
        config = ConfigEval(
            modelo=modelo,
            epochs=args.epochs,
            k_folds=args.k_folds,
            paciente=paciente,
        )

    # Ejecución por fold
    else:
        config = ConfigEval(
            modelo=modelo,
            epochs=args.epochs,
            k_folds=args.k_folds,
            fold_test=args.fold_test,
        )

    ejecutar_flujo_eval(config=config, limpiar=args.limpiar, verbose=True)


def ejecutar_eval_pipeline(
    modelo,
    paciente=None,
    fold_test=None,
    plano=None,
    epochs=50,
    k_folds=5,
    limpiar=False,
):
    """
    Entrada interna para el pipeline: recibe objetos ya
    construidos y ejecuta el flujo sin usar el parser CLI.
    """
    # Ejecución por paciente
    if paciente is not None:
        config = ConfigEval(
            modelo=modelo,
            epochs=epochs,
            k_folds=k_folds,
            paciente=paciente,
            plano_forzado=plano,
        )

    # Ejecución por fold
    elif fold_test is not None:
        config = ConfigEval(
            modelo=modelo,
            epochs=epochs,
            k_folds=k_folds,
            fold_test=fold_test,
            plano_forzado=plano,
        )

    else:
        raise ValueError("Debe especificarse un paciente o un fold de test.")

    ejecutar_flujo_eval(
        config=config,
        limpiar=limpiar,
    )


if __name__ == "__main__":
    main()
