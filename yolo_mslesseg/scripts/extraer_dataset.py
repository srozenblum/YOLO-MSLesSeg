"""
Script: extraer_dataset.py

Descripción:
    Genera el dataset YOLO anotado para el flujo de trabajo a partir del dataset de
    entrada MSLesSeg, ya sea para un paciente individual o para todos los pacientes.
    Puede ejecutarse tanto de forma independiente (desde CLI) como internamente
    dentro del pipeline (`ejecutar_pipeline.py`).

    Extrae cortes bidimensionales de imágenes de resonancia magnética a partir
    de las modalidades seleccionadas (T1, T2, FLAIR) y del plano anatómico
    especificado, generando automáticamente la estructura esperada por YOLO
    (images/, GT_masks/, labels/) y dividiendo el dataset YOLO en folds para validación
    cruzada. Estos conjuntos se utilizan posteriormente en las etapas de entrenamiento,
    predicción, reconstrucción, consenso y evaluación.

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
        Número de cortes a extraer (valor entero o percentil, por ejemplo 50 o 'P75').

    --mejora (str, opcional)
        Algoritmo de mejora de imagen ('HE', 'CLAHE', 'GC', 'LT', o None).
        Por defecto None.

    --k_folds (int, opcional)
        Número de folds para validación cruzada.
        Por defecto 5.

    --completo (flag, excluyente con --paciente_id)
        Generar el dataset YOLO para todos los pacientes.

    --paciente_id (str, excluyente con --completo)
        Generar el dataset YOLO solo para el paciente indicado (ejemplo: 'P12').

    --limpiar (flag, opcional)
        Limpiar el dataset YOLO previo antes de extraer uno nuevo.

Uso por CLI:
        python -m yolo_mslesseg.scripts.extraer_dataset \
        --plano "sagital" \
        --modalidad "T1" \
        --num_cortes 100 \
        --epochs 50 \
        --k_folds 4 \
        --completo \

Entradas:
    - Dataset: MSLesSeg-Dataset/train/
        Contiene los volúmenes originales de resonancia magnética y sus máscaras asociadas.

    - Clases:
        * ConfigDataset → gestiona directorios y variables globales para la extracción del dataset YOLO.
        * Modelo → define el plano, modalidades, mejora y número de cortes del experimento.

Salidas:
    - Estructura YOLO con imágenes, máscaras GT y etiquetas en formato .txt.
"""

import argparse
import logging
import sys

import numpy as np
from matplotlib import pyplot as plt
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg
from ultralytics.utils import LOGGER

from yolo_mslesseg.configs.ConfigDataset import ConfigDataset
from yolo_mslesseg.utils.Modelo import Modelo
from yolo_mslesseg.utils.Paciente import Paciente
from yolo_mslesseg.utils.configurar_logging import get_logger
from yolo_mslesseg.utils.utils import (
    listar_pacientes,
    normalizar_mascara_binaria,
    verificar_grises,
    evaluar_resultados,
    int_o_percentil,
    calcular_fold,
)

# Configurar logger
logger = get_logger(__file__)

# Ocultar loggings de ultralytics
LOGGER.setLevel(logging.WARNING)


# =============================
#      FUNCIONES AUXILIARES
# =============================


def calcular_num_cortes_percentil(input_dir, plano, modalidad, percentil=50):
    """
    Calcula el número de cortes a usar en base al percentil global
    de la distribución de cortes con lesión en todos los pacientes.
    """
    pacientes = listar_pacientes(input_dir)

    lista_cortes = []

    for paciente_id in pacientes:
        paciente = Paciente(id=paciente_id, plano=plano, modalidad=modalidad)
        indices = paciente.indices_a_usar()  # Todos los cortes con lesión
        lista_cortes.append(len(indices))

    if not lista_cortes:
        raise ValueError(
            f"No se encontraron cortes con lesión válidos para calcular el percentil en {input_dir}."
        )

    # Calcular percentil global
    try:
        num_cortes = int(np.percentile(lista_cortes, percentil))
    except Exception as e:
        raise ValueError(f"Percentil no válido ({percentil}): {e}")

    return num_cortes


def resolver_num_cortes(num_cortes, input_dir, plano, modalidad):
    """
    Resuelve el número de cortes a usar según valor fijo o percentil,
    devolviendo una tupla (num_cortes, percentil).
    """
    # Num_cortes es número entero
    if isinstance(num_cortes, int) or num_cortes is None:
        return num_cortes, None

    # Num_cortes es percentil
    elif isinstance(num_cortes, str) and num_cortes.startswith("P"):
        percentil = int(num_cortes[1:])
        num_cortes_percentil = calcular_num_cortes_percentil(
            input_dir=input_dir, plano=plano, modalidad=modalidad, percentil=percentil
        )
        return num_cortes_percentil, percentil

    else:
        raise ValueError(f"Formato de num_cortes no válido: {num_cortes}.")


def construir_paths(paciente, config):
    """
    Construye un diccionario de paths (images, GT_masks, labels)
    para un paciente individual dentro del fold correspondiente.
    """
    fold = calcular_fold(paciente_id=paciente.id, k_folds=config.k_folds)
    root = config.output_dir / f"fold{fold}" / paciente.id / paciente.plano

    return {
        "images": root / "images",
        "GT_masks": root / "GT_masks",
        "labels": root / "labels",
    }


def guardar_cortes(paciente, images_dir, gt_masks_dir, num_cortes):
    """
    Guarda los cortes de imagen y máscara correspondientes
    a un paciente en los directorios indicados.
    """
    cortes_img = paciente.cortes_con_lesion_img(num_cortes=num_cortes)
    cortes_gt_mask = paciente.cortes_con_lesion_mask(num_cortes=num_cortes)

    if not cortes_img or not cortes_gt_mask:
        raise ValueError(
            f"No se encontraron cortes válidos para el paciente {paciente.id}."
        )

    # Imágenes
    for modalidad, lista_cortes in cortes_img.items():
        for i, corte in lista_cortes:
            corte_gray = verificar_grises(corte)  # Asegurar escala de grises
            img_path = images_dir / f"{paciente.id}_{modalidad}_{i}.png"
            plt.imsave(img_path, corte_gray.T, cmap="gray", origin="lower")

    # Máscaras
    for i, corte in cortes_gt_mask:
        gt_mask_path = gt_masks_dir / f"{paciente.id}_{i}.png"
        plt.imsave(gt_mask_path, corte.T, cmap="gray", origin="lower")


def normalizar_mascaras(gt_masks_dir):
    """
    Normaliza todas las máscaras en GT_masks_dir a valores binarios (0 y 1).
    """
    archivos = list(gt_masks_dir.glob("*.png"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron máscaras .png en {gt_masks_dir}")

    for path in archivos:
        try:
            normalizar_mascara_binaria(path)
        except Exception as e:
            raise OSError(f"Error al normalizar {path.name}: {e}")


def anotar_mascaras(gt_masks_dir, labels_dir):
    """
    Convierte las máscaras GT de un paciente a
    anotaciones bajo el formato YOLO.
    """
    # Normalizar máscaras antes de la conversión
    normalizar_mascaras(gt_masks_dir)

    convert_segment_masks_to_yolo_seg(
        masks_dir=gt_masks_dir,
        output_dir=labels_dir,
        classes=1,
    )


# =============================
#        PROCESAMIENTO
# =============================


def procesar_paciente_dataset(paciente, config, paths_dir=None, num_cortes="P50"):
    """
    Ejecuta el proceso de extracción de cortes para un paciente individual.
    """
    # Si no se pasan los directorios → modo paciente → asigna los del config
    if paths_dir is None:
        paths_dir = config.paciente_dir

    if all(path.is_dir() and any(path.iterdir()) for path in paths_dir.values()):
        return  # Dataset ya existente

    guardar_cortes(
        paciente=paciente,
        images_dir=paths_dir["images"],
        gt_masks_dir=paths_dir["GT_masks"],
        num_cortes=num_cortes,
    )

    anotar_mascaras(gt_masks_dir=paths_dir["GT_masks"], labels_dir=paths_dir["labels"])

    return True


def guardar_cortes_por_paciente(input_dir, config, num_cortes):
    """
    Ejecuta el proceso de extracción de cortes para todos los pacientes en input_dir.
    """
    pacientes = listar_pacientes(input_dir)

    resultados = []
    for paciente_id in pacientes:
        paciente = Paciente(
            id=paciente_id,
            plano=config.modelo.plano,
            modalidad=config.modelo.modalidad,
            mejora=config.modelo.mejora,
        )
        paths_dir = construir_paths(paciente=paciente, config=config)
        try:
            paciente_procesado = procesar_paciente_dataset(
                paciente=paciente,
                config=config,
                paths_dir=paths_dir,
                num_cortes=num_cortes,
            )
            resultados.append(paciente_procesado)
        except Exception as e:
            logger.warning(
                f"⚠️ Error extrayendo dataset YOLO de {paciente_id}, se omite: {e}."
            )
            continue

    return evaluar_resultados(resultados)


# =============================
#       FLUJO PRINCIPAL
# =============================


def ejecutar_flujo_dataset(config, limpiar, verbose=False):
    """
    Ejecuta el flujo principal de extracción del dataset YOLO
    ya sea sobre todos los pacientes o sobre un paciente individual.
    """
    if verbose:
        str_completo = "conjunto de pacientes completo"
        str_paciente = f"paciente {config.paciente}"
        logger.header(
            f"\n🧩 Preparando dataset YOLO para el {str_paciente if config.es_paciente_individual else str_completo}."
        )

    # Limpiar si corresponde
    if limpiar:
        if verbose:
            logger.info(f"♻️ Limpiando dataset YOLO previo.")
        config.limpiar_dataset()

    # Verificar paths
    config.verificar_paths()

    # Calcular número de cortes
    num_cortes, percentil = resolver_num_cortes(
        num_cortes=config.modelo.num_cortes,
        input_dir=config.dataset_entrada,
        plano=config.modelo.plano,
        modalidad=config.modelo.modalidad,
    )

    if percentil is None:
        logger.info(f"📊 Número de cortes a extraer: {num_cortes}.")
    else:
        logger.info(f"📊 Número de cortes a extraer: {num_cortes} (P{percentil}).")

    # Ejecución por paciente
    if config.es_paciente_individual:
        dataset_extraido = procesar_paciente_dataset(
            paciente=config.paciente, config=config, num_cortes=num_cortes
        )
        if dataset_extraido is None:
            logger.skip(f"⏩ Dataset YOLO ya existente.")

        elif dataset_extraido is True:
            logger.info(f"✅ Extracción de cortes completada.")
            logger.info(f"📝 Anotaciones completadas.")

        else:
            logger.warning(f"⚠️ Estado desconocido al extraer el dataset YOLO.")

    # Ejecucion por fold
    else:
        procesados = guardar_cortes_por_paciente(
            input_dir=config.dataset_entrada, config=config, num_cortes=num_cortes
        )

        if procesados is None:  # Ningún paciente procesado
            logger.skip(f"⏩ Dataset YOLO ya existente.")
        elif (
            procesados is True
        ):  # Todos los pacientes procesados → dataset creado → dividir en folds
            logger.info(f"🆗 Dataset YOLO extraído con éxito.")
        elif (
            procesados == "parcial"
        ):  # Algunos pacientes procesados → dataset actualizado → dividir en folds
            logger.info(f"🔁 Dataset YOLO parcialmente actualizado.")
        else:
            logger.warning("⚠️ Estado desconocido al extraer el dataset YOLO.")


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
        description="Extraer un dataset YOLO para flujo de trabajo YOLO-MSLesSeg.",
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
        help="Número de cortes a extraer (valor entero o percentil).",
    )
    parser.add_argument(
        "--mejora",
        type=str,
        default=None,
        choices=["HE", "CLAHE", "GC", "LT"],
        metavar="[HE, CLAHE, GC, LT]",
        help="Algoritmo de mejora de imagen a aplicar. Por defecto None.",
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
        "--completo",
        action="store_true",
        help="Extraer el dataset YOLO para todos los pacientes, "
        "dividiéndolos en <k_folds> para validación cruzada.",
    )
    group.add_argument(
        "--paciente_id",
        type=str,
        metavar="<paciente_id>",
        help="Extraer el dataset YOLO solo para el paciente indicado.",
    )
    parser.add_argument(
        "--limpiar",
        action="store_true",
        default=False,
        help="Limpiar el dataset YOLO extraído previamente.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    """
    Entrada CLI del script: parsea argumentos, construye Modelo/Paciente/ConfigDataset
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
        config = ConfigDataset(
            modelo=modelo,
            k_folds=args.k_folds,
            paciente=paciente,
        )

    # Ejecución completa
    else:
        config = ConfigDataset(
            modelo=modelo,
            k_folds=args.k_folds,
            completo=True,
        )

    ejecutar_flujo_dataset(
        config=config,
        limpiar=args.limpiar,
        verbose=True,
    )


def ejecutar_dataset_pipeline(modelo, paciente=None, k_folds=5, limpiar=False):
    """
    Entrada interna para el pipeline: recibe objetos ya
    construidos y ejecuta el flujo sin usar el parser CLI.
    """
    # Ejecución por paciente
    if paciente is not None:
        config = ConfigDataset(
            modelo=modelo,
            k_folds=k_folds,
            paciente=paciente,
        )

    # Ejecución completa
    else:
        config = ConfigDataset(
            modelo=modelo,
            k_folds=k_folds,
            completo=True,
        )

    ejecutar_flujo_dataset(
        config=config,
        limpiar=limpiar,
    )


if __name__ == "__main__":
    main()
