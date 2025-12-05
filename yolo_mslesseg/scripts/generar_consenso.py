"""
Script: generar_consenso.py

Descripción:
    Combina las predicciones volumétricas obtenidas desde los tres planos anatómicos
    (axial, coronal y sagital) para generar un volumen de consenso 3D en formato NIfTI.
    El consenso se calcula mediante votación mayoritaria (umbral ≥2 o 3) y se valida
    automáticamente frente al ground truth. Puede ejecutarse a nivel de paciente o de fold.

Modos de ejecución:
    1. CLI (uso independiente):
       - Se leen y parsean los argumentos escritos por el usuario en la línea de comandos.
       - Se crean las instancias de Modelo y, opcionalmente, Paciente.

    2. Interno (desde `ejecutar_pipeline.py`):
       - Se reciben instancias ya creadas de Modelo y (opcionalmente) Paciente,
       junto con el resto de parámetros.
       - No se usa el parser de argumentos.

Argumentos CLI:
    --modalidad (list[str], opcional)
        Modalidad o modalidades de imagen ('T1', 'T2', 'FLAIR').
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

    --umbral (int, opcional)
        Umbral de votación para consenso (2 = mayoría simple, 3 = unanimidad).
        Por defecto 2.

    --fold_test (int, excluyente con --paciente_id)
        Generar el consenso para todos los pacientes del
        fold indicado, correspondiente al conjunto de test.

    --paciente_id (str, excluyente con --fold_test)
        Generar el consenso solo para el paciente indicado.

    --limpiar (flag, opcional)
        Limpia los consensos previos antes de generar nuevos.

Uso por CLI:
    python -m yolo_mslesseg.scripts.generar_consenso \
        --epochs 50 \
        --num_cortes 20 \
        --k_folds 5 \
        --fold_test 1 \

Entradas:
    - Volúmenes predichos (.nii.gz): generados previamente por `reconstruir_volumen.py` en
        almacenados en vols/<mejora>/<modalidad>_<num_cortes>c_<k_folds>folds_<epochs>epochs/<fold_test>/PX/

    - Ground truth (.nii.gz): volúmenes originales ubicados en GT/<paciente_id>/
        utilizados como referencia para la validación.

    - Clases:
        * ConfigConsenso → gestiona rutas y variables globales para la generación de consensos.
        * Modelo → define la modalidad, número de cortes, mejora y configuración del experimento.

Salidas:
    - Volúmenes de consenso 3D (.nii.gz) en
        vols/<mejora>/<modalidad>_<num_cortes>c_<k_folds>folds_<epochs>epochs/<fold_test>/PX/
"""

import argparse
import sys

import numpy as np

from yolo_mslesseg.configs.ConfigConsenso import ConfigConsenso
from yolo_mslesseg.utils.Modelo import Modelo
from yolo_mslesseg.utils.Paciente import Paciente
from yolo_mslesseg.utils.configurar_logging import get_logger
from yolo_mslesseg.utils.utils import (
    cargar_volumen,
    guardar_volumen,
    int_o_percentil,
    ruta_existente,
    listar_pacientes,
    cargar_referencia_nifti,
    evaluar_resultados,
    reconstruccion_valida,
    log_estado_fold,
)

# Configurar logger
logger = get_logger(__file__)


# =============================
#         FUNCIONES BASE
# =============================


def combinar_volumenes(axial_vol, coronal_vol, sagital_vol, umbral=2):
    """Devuelve un volumen binario de consenso combinando tres volúmenes según el umbral establecido."""
    consenso = ((axial_vol + coronal_vol + sagital_vol) >= umbral).astype(np.uint8)
    return consenso


def generar_consenso(axial_path, coronal_path, sagital_path, output_path, umbral=2):
    """Genera y guarda el volumen consenso a partir de los archivos NIfTI de los tres planos."""
    axial_vol = cargar_volumen(axial_path)
    coronal_vol = cargar_volumen(coronal_path)
    sagital_vol = cargar_volumen(sagital_path)
    affine = cargar_referencia_nifti(axial_path)[1]

    # Combinar los volúmenes aplicando el umbral para generar el consenso
    consenso = combinar_volumenes(
        axial_vol=axial_vol,
        coronal_vol=coronal_vol,
        sagital_vol=sagital_vol,
        umbral=umbral,
    )

    guardar_volumen(volumen=consenso, affine=affine, output_path=output_path)


# =============================
#        PROCESAMIENTO
# =============================


def procesar_paciente_consenso(
    config,
    paths_dir=None,
    umbral=2,
):
    """Ejecuta el proceso de generación de consenso para un paciente individual."""
    # Si no se pasan los directorios → modo paciente → asigna los del config
    if paths_dir is None:
        paths_dir = config.paciente_pred_vol
        gt_vol = config.paciente_gt_vol
    else:
        gt_vol = paths_dir["gt"]

    # Evitar reprocesar si ya existe el consenso
    if ruta_existente(paths_dir["consenso"]):
        return

    generar_consenso(
        axial_path=paths_dir["axial"],
        coronal_path=paths_dir["coronal"],
        sagital_path=paths_dir["sagital"],
        output_path=paths_dir["consenso"],
        umbral=umbral,
    )

    if not reconstruccion_valida(paths_dir["consenso"], gt_vol):
        raise RuntimeError("Reconstrucción de consenso no válida.")

    return True


def construir_paths(paciente_id, config):
    """
    Construye un diccionario de paths (axial, coronal, sagital, ground truth)
    para un paciente individual.
    """
    paths = {
        plano: config.vols_fold_dir / paciente_id / f"{paciente_id}_{plano}.nii.gz"
        for plano in config.PLANOS
    }
    paths["gt"] = config.gt_dir / paciente_id / f"{paciente_id}_MASK.nii.gz"

    return paths


def generar_consenso_por_paciente(input_dir, config, umbral=2):
    """Ejecuta el proceso de generación de consenso para todos los pacientes en input_dir."""
    pacientes = listar_pacientes(input_dir)
    resultados = []

    for paciente_id in pacientes:
        paths_paciente = construir_paths(paciente_id, config)
        try:
            consenso = procesar_paciente_consenso(
                config=config,
                paths_dir=paths_paciente,
                umbral=umbral,
            )
            resultados.append(consenso)
        except Exception as e:
            logger.warning(
                f"⚠️ Error generando consenso de {paciente_id}, se omite: {e}."
            )
            continue

    return evaluar_resultados(resultados)


# =============================
#       FLUJO PRINCIPAL
# =============================


def ejecutar_flujo_consenso(config, limpiar, umbral=2, verbose=False):
    """
    Ejecuta el flujo principal de generación de consenso,
    ya sea sobre un fold o un paciente individual.
    """
    if verbose:
        str_fold = f"fold {config.fold_test}"
        str_paciente = f"paciente {config.paciente}"
        logger.header(
            f"\n🤝 Generando consenso para el {str_paciente if config.es_paciente_individual else str_fold}."
        )

    # Limpiar si corresponde
    if limpiar:
        if verbose:
            logger.info("♻️ Limpiando consensos previos.")
        config.limpiar_consenso()

    # Verificar paths
    config.verificar_paths()

    # Ejecución por paciente
    if config.es_paciente_individual:
        consenso_generado = procesar_paciente_consenso(config=config, umbral=umbral)
        if consenso_generado is None:
            logger.skip(f"⏩ Consenso generado ya existente.")
        elif consenso_generado is True:
            logger.info(f"✅ Consenso generado correctamente.")
        else:
            logger.warning(f"⚠️ Estado desconocido al generar consenso.")

    # Ejecución por fold
    else:
        consensos_generados = generar_consenso_por_paciente(
            input_dir=config.vols_fold_dir, config=config, umbral=umbral
        )
        log_estado_fold(
            logger=logger, resultado=consensos_generados, fold=config.fold_test
        )


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
        description="Generar una máscara de consenso mediante votación mayoritaria.",
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
        "--umbral",
        type=int,
        default=2,
        choices=[2, 3],
        metavar="",
        help="Umbral para generar el consenso. Por defecto 2.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--fold_test",
        type=int,
        metavar="<fold_test>",
        help="Generar los consensos para el fold indicado, utilizado como conjunto de test.",
    )
    group.add_argument(
        "--paciente_id",
        type=str,
        metavar="<paciente_id>",
        help="Generar el consenso solo para el paciente indicado.",
    )
    parser.add_argument(
        "--limpiar",
        action="store_true",
        default=False,
        help="Limpia el consenso generado previamente.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    """
    Entrada CLI del script: parsea argumentos, construye Modelo/Paciente/ConfigConsenso
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
        config = ConfigConsenso(
            modelo=modelo,
            epochs=args.epochs,
            k_folds=args.k_folds,
            paciente=paciente,
        )

    # Ejecución por fold
    else:
        config = ConfigConsenso(
            modelo=modelo,
            epochs=args.epochs,
            k_folds=args.k_folds,
            fold_test=args.fold_test,
        )

    ejecutar_flujo_consenso(
        config=config, umbral=args.umbral, limpiar=args.limpiar, verbose=True
    )


def ejecutar_consenso_pipeline(
    modelo, paciente=None, fold_test=None, epochs=50, k_folds=5, umbral=2, limpiar=False
):
    """
    Entrada interna para el pipeline: recibe objetos ya
    construidos y ejecuta el flujo sin usar el parser CLI.
    """
    # Ejecución por paciente
    if paciente is not None:
        config = ConfigConsenso(
            modelo=modelo,
            epochs=epochs,
            k_folds=k_folds,
            paciente=paciente,
        )

    # Ejecución por fold
    elif fold_test is not None:
        config = ConfigConsenso(
            modelo=modelo,
            epochs=epochs,
            k_folds=k_folds,
            fold_test=fold_test,
        )

    else:
        raise ValueError("Debe especificarse un paciente o un fold de test.")

    ejecutar_flujo_consenso(
        config=config,
        umbral=umbral,
        limpiar=limpiar,
    )


if __name__ == "__main__":
    main()
