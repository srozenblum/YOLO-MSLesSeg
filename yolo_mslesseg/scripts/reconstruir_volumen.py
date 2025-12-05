"""
Script: reconstruir_volumen.py

Descripción:
    Reconstruye los volúmenes 3D en formato NIfTI a partir de las máscaras
    de predicción 2D generadas por el modelo YOLO, ya sea para un paciente
    individual o para todos los pacientes de un fold. Los volúmenes
    reconstruidos se validan automáticamente frente a las máscaras ground
    truth y se almacenan en el directorio vols/.

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
        Plano anatómico del modelo ('axial', 'coronal', 'sagital').

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

    --fold_test (int, excluyente con --paciente_id)
        Reconstruir los volúmenes para todos los pacientes del
        fold indicado, correspondiente al conjunto de test.

    --paciente_id (str, excluyente con --fold_test)
        Reconstruir los volúmenes solo para el paciente indicado.

    --limpiar (flag, opcional)
        Limpia el directorio de volúmenes predichos antes de generar nuevas reconstrucciones.

Uso:
    python -m yolo_mslesseg.scripts.reconstruir_volumen \
        --plano "sagital" \
        --num_cortes P75 \
        --epochs 25 \
        --fold_test 1 \

Entradas:
    - Máscaras predichas 2D (.png): generadas previamente con `generar_predicciones.py`,
        en el subdirectorio pred_masks/ dentro de cada paciente.

    - Ground truth (.nii.gz): volúmenes originales ubicados en GT/<paciente_id>/
        utilizados como referencia para la reconstrucción y validación.

    - Clases:
        * ConfigRecVol → gestiona rutas y variables globales asociadas a la reconstrucción.
        * Modelo → define el plano, modalidades, mejora y número de cortes del experimento.

Salidas:
    - Volúmenes reconstruidos (.nii.gz) en
    vols/<mejora>/<modalidad>_<num_cortes>c_<k_folds>folds_<epochs>epochs/<fold_test>/PX/
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from yolo_mslesseg.configs.ConfigRecVol import ConfigRecVol
from yolo_mslesseg.utils.Modelo import Modelo
from yolo_mslesseg.utils.Paciente import Paciente
from yolo_mslesseg.utils.configurar_logging import get_logger
from yolo_mslesseg.utils.utils import (
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


def extraer_indices_png(input_dir):
    """
    Extrae los índices numéricos de los archivos PNG en input_dir,
    devolviendo una lista [(nombre_archivo, índice)] ordenada por índice.
    """
    if not ruta_existente(input_dir):
        raise FileNotFoundError(
            f"No se encontró el directorio de máscaras predichas: {input_dir}"
        )

    patron = re.compile(r".*_(\d+)(?:_[^_]*)?\.png$", re.IGNORECASE)
    tuplas = []

    for p in input_dir.glob("*.png"):
        m = patron.match(p.name)
        if m:
            tuplas.append((p.name, int(m.group(1))))
        else:
            logger.warning(f"⚠️ No se pudo extraer el índice de {p.name}")

    if not tuplas:
        raise FileNotFoundError("No hay máscaras predichas que procesar.")

    # Ordenar por índice
    tuplas.sort(key=lambda t: t[1])
    return tuplas


def cargar_y_preprocesar_imagen(img_path):
    """Carga una imagen PNG, la convierte a escala de grises y la binariza si es necesario."""
    if not ruta_existente(img_path):
        raise FileNotFoundError(f"No se encontró la imagen: {img_path}")

    img = Image.open(img_path)
    img_array = np.array(img)

    if len(img_array.shape) > 2:  # RGB → gris
        img_array = img_array[:, :, 0]

    if np.max(img_array) > 1:  # Binarizar si no está normalizada
        img_array = (img_array > 0).astype(np.float32)

    return img_array


def validar_corte(indice, img_array, shape_original, plano):
    """
    Verifica que el índice y las dimensiones del corte sean consistentes con el volumen original.
    """
    # 1. Validar índice dentro de rango
    max_indices = {
        "axial": shape_original[2],
        "coronal": shape_original[1],
        "sagital": shape_original[0],
    }
    if indice < 0 or indice >= max_indices[plano]:
        raise ValueError(f"Índice {indice} fuera de rango para plano {plano}.")

    # 2. Validar dimensiones del corte
    expected_shapes = {
        "axial": (shape_original[0], shape_original[1]),
        "coronal": (shape_original[0], shape_original[2]),
        "sagital": (shape_original[1], shape_original[2]),
    }
    if img_array.shape != expected_shapes[plano]:
        raise ValueError(
            f"Dimensiones {img_array.shape} incorrectas para plano {plano}. "
            f"Se esperaba {expected_shapes[plano]}."
        )


def insertar_corte(volumen, img_array, indice, plano):
    """Inserta un corte 2D en el volumen según el plano correspondiente."""
    if plano == "axial":
        volumen[:, :, indice] = img_array
    elif plano == "coronal":
        volumen[:, indice, :] = img_array
    elif plano == "sagital":
        volumen[indice, :, :] = img_array


# =============================
#       RECONSTRUCCIÓN 3D
# =============================


def necesita_reconstruccion(pred_path):
    """Devuelve True si el volumen reconstruido no existe o está vacío."""
    return not ruta_existente(pred_path) or pred_path.stat().st_size == 0


def reconstruir_volumen(pred_masks_dir, volumen_referencia, output_path, plano):
    """Reconstruye un volumen 3D a partir de máscaras predichas 2D."""
    shape_original, affine = cargar_referencia_nifti(volumen_referencia)
    volumen = np.zeros(shape_original, dtype=np.float32)
    indices = extraer_indices_png(pred_masks_dir)

    for archivo, indice in indices:
        pred_mask_path = Path(pred_masks_dir) / archivo
        pred_mask_array = cargar_y_preprocesar_imagen(pred_mask_path)

        validar_corte(indice, pred_mask_array, shape_original, plano)
        insertar_corte(volumen, pred_mask_array, indice, plano)

    guardar_volumen(volumen, affine, output_path)
    return volumen


# =============================
#        PROCESAMIENTO
# =============================


def procesar_paciente_vol(paciente_id, config, paths_dir=None):
    """
    Ejecuta el proceso de reconstrucción de volumen
    (verificación → reconstrucción → validación)
    para un paciente individual.
    """
    # Si no se pasan los directorios → modo paciente → asigna los del config
    if paths_dir is None:
        paths_dir = {
            "pred_vol": config.paciente_pred_vol,
            "gt_vol": config.paciente_gt_vol,
            "pred_masks": config.paciente_pred_masks,
        }

    pred_vol_paciente = paths_dir["pred_vol"]
    gt_vol_paciente = paths_dir["gt_vol"]
    pred_masks_paciente = paths_dir["pred_masks"]

    # Comprobar si ya existe un volumen reconstruido
    if not necesita_reconstruccion(pred_vol_paciente):

        # Si existe, validar que sea consistente con el ground truth
        if not reconstruccion_valida(
            pred_vol_path=pred_vol_paciente,
            gt_vol_path=gt_vol_paciente,
        ):
            # Si es inválido, marcar para reconstrucción
            logger.warning(
                f"⚠️ Reconstrucción inválida para {paciente_id}, debe rehacerse."
            )
            reconstruccion_necesaria = True
        else:
            # Si es válido, no hace falta reconstruir
            return
    else:
        # Si no existe el volumen, reconstruir
        reconstruccion_necesaria = True

    # Reconstruir y validar el resultado
    if reconstruccion_necesaria:
        reconstruir_volumen(
            pred_masks_dir=pred_masks_paciente,
            volumen_referencia=gt_vol_paciente,
            output_path=pred_vol_paciente,
            plano=config.modelo.plano,
        )
        if not reconstruccion_valida(
            pred_vol_path=pred_vol_paciente, gt_vol_path=gt_vol_paciente
        ):
            raise RuntimeError(f"Validación fallida para {paciente_id}.")
        return True


def construir_paths(paciente_id, config):
    """
    Construye un diccionario de paths (pred_vol, gt_vol, pred_masks)
    para un paciente individual.
    """
    root_vols = config.vols_fold_dir / paciente_id
    root_gt = config.gt_dir / paciente_id
    root_dataset = config.dataset_fold_dir / paciente_id / config.plano

    return {
        "pred_vol": root_vols / f"{paciente_id}_{config.plano}.nii.gz",
        "gt_vol": root_gt / f"{paciente_id}_MASK.nii.gz",
        "pred_masks": root_dataset / "pred_masks",
    }


def reconstruir_volumen_por_paciente(input_dir, config):
    """Ejecuta el proceso de reconstrucción de volumen para todos los pacientes en input_dir."""
    pacientes = listar_pacientes(input_dir)

    resultados = []
    for paciente_id in pacientes:
        paths_paciente = construir_paths(paciente_id, config)
        try:
            volumen_reconstruido = procesar_paciente_vol(
                paciente_id=paciente_id, config=config, paths_dir=paths_paciente
            )
            resultados.append(volumen_reconstruido)
        except Exception as e:
            logger.warning(
                f"⚠️ Error reconstruyendo volúmenes de {paciente_id}, se omite: {e}.",
            )
            continue

    return evaluar_resultados(resultados)


# =============================
#       FLUJO PRINCIPAL
# =============================


def ejecutar_flujo_vol(config, limpiar, verbose=False):
    """
    Ejecuta el flujo principal de reconstrucción,
    ya sea sobre un fold o un paciente individual.
    """

    if verbose:
        str_fold = f"fold {config.fold_test}"
        str_paciente = f"paciente {config.paciente}"
        logger.header(
            f"\n🧱 Reconstruyendo volumen para el {str_paciente if config.es_paciente_individual else str_fold}."
        )

    # Limpiar si corresponde
    if limpiar:
        if verbose:
            logger.info(f"♻️ Limpiando volúmenes previos.")
        config.limpiar_volumenes()

    # Verificar paths
    config.verificar_paths()

    # Ejecución por paciente
    if config.es_paciente_individual:
        paciente_id = config.paciente.id
        volumen_reconstruido = procesar_paciente_vol(
            paciente_id=paciente_id, config=config
        )
        if volumen_reconstruido is None:
            logger.skip(f"⏩ Volumen reconstruido ya existente.")
        elif volumen_reconstruido is True:
            logger.info(f"✅ Volumen reconstruido y validado con éxito.")
        else:
            logger.warning(f"⚠️ Estado desconocido al reconstruir volumen.")

    # Ejecucion por fold
    else:
        reconstruidos = reconstruir_volumen_por_paciente(
            input_dir=config.dataset_fold_dir, config=config
        )
        log_estado_fold(logger=logger, resultado=reconstruidos, fold=config.fold_test)


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
        description="Aplicar un modelo YOLO entrenado para generar máscaras 2D de predicción.",
    )
    parser.add_argument(
        "--plano",
        type=str,
        required=True,
        choices=["axial", "coronal", "sagital"],
        metavar="[axial, coronal, sagital]",
        help="Plano anatómico del modelo.",
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
        help="Reconstruir los volúmenes para el fold indicado, utilizado como conjunto de test.",
    )
    group.add_argument(
        "--paciente_id",
        type=str,
        metavar="<paciente_id>",
        help="Reconstruir los volúmenes solo para el paciente indicado.",
    )
    parser.add_argument(
        "--limpiar",
        action="store_true",
        default=False,
        help="Limpiar los volúmenes reconstruidos previamente.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    """
    Entrada CLI del script: parsea argumentos, construye Modelo/Paciente/ConfigRecVol
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
        config = ConfigRecVol(
            modelo=modelo,
            epochs=args.epochs,
            k_folds=args.k_folds,
            paciente=paciente,
        )

    # Ejecución por fold
    else:
        config = ConfigRecVol(
            modelo=modelo,
            epochs=args.epochs,
            k_folds=args.k_folds,
            fold_test=args.fold_test,
        )

    ejecutar_flujo_vol(config=config, limpiar=args.limpiar, verbose=True)


def ejecutar_reconstrucciones_pipeline(
    modelo, paciente=None, fold_test=None, epochs=50, k_folds=5, limpiar=False
):
    """
    Entrada interna para el pipeline: recibe objetos ya
    construidos y ejecuta el flujo sin usar el parser CLI.
    """
    # Ejecución por paciente
    if paciente is not None:
        config = ConfigRecVol(
            modelo=modelo,
            epochs=epochs,
            k_folds=k_folds,
            paciente=paciente,
        )

    # Ejecución por fold
    elif fold_test is not None:
        config = ConfigRecVol(
            modelo=modelo,
            epochs=epochs,
            k_folds=k_folds,
            fold_test=fold_test,
        )

    else:
        raise ValueError("Debe especificarse un paciente o un fold de test.")

    ejecutar_flujo_vol(
        config=config,
        limpiar=limpiar,
    )


if __name__ == "__main__":
    main()
