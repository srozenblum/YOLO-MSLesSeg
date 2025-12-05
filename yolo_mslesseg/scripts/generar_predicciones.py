"""
Script: generar_predicciones.py

Descripción:
    Aplica un modelo YOLO entrenado para generar las máscaras de predicción
    2D correspondientes a cada corte del conjunto de test, ya sea para un
    paciente individual o para todos los pacientes de un fold. Puede ejecutarse
    tanto de forma independiente (desde CLI) como internamente dentro del
    pipeline (`ejecutar_pipeline.py`).

    Las máscaras generadas se almacenan en el subdirectorio pred_masks/
    dentro de cada paciente, manteniendo la estructura del dataset original.

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
         Generar las predicciones 2D para todos los pacientes del
         fold indicado, correspondiente al conjunto de test.

    --paciente_id (str, excluyente con --fold_test)
        Generar las predicciones 2D solo para el paciente indicado.

    --limpiar (flag, opcional)
        Limpiar el directorio con las predicciones 2D binarias antes de generar nuevas.

Uso por CLI:
    python -m yolo_mslesseg.scripts.generar_predicciones \
        --plano "coronal" \
        --modalidad "FLAIR" \
        --num_cortes 50 \
        --epochs 100 \
        --fold_test 3 \

Entradas:
    - Dataset: generado previamente con extraer_dataset.py y entrenado con train.py
        Contiene las imágenes de entrada y sus anotaciones YOLO divididas por folds.

    - Pesos del modelo: archivo best.pt
        Archivo de pesos del modelo YOLO entrenado, ubicado en el directorio
        trains/<mejora>/<modalidad>_<num_cortes>c_<k_folds>folds_<epochs>epochs/
        <plano>/<fold_test>/weights/best.pt.

    - Clases:
        * ConfigPred → gestiona directorios y variables globales asociados a la generación de predicciones.
        * Modelo → define el plano, modalidades, mejora y número de cortes del experimento.

Salidas:
    - Máscaras de predicción 2D (.png) en el subdirectorio del paciente pred_masks/.
"""

import argparse
import sys

import cv2
import numpy as np
from tqdm import tqdm

from yolo_mslesseg.configs.ConfigPred import ConfigPred
from yolo_mslesseg.utils.Modelo import Modelo
from yolo_mslesseg.utils.Paciente import Paciente
from yolo_mslesseg.utils.configurar_logging import get_logger
from yolo_mslesseg.utils.utils import (
    listar_pacientes,
    int_o_percentil,
    ruta_existente,
    evaluar_resultados,
    crear_directorio,
    cargar_modelo,
    log_estado_fold,
)

# Configurar logger
logger = get_logger(__file__)


# =============================
#        FUNCIONES BASE
# =============================


def ejecutar_prediccion(modelo, img_array):
    """Ejecuta el modelo YOLO sobre una imagen y devuelve las máscaras crudas."""
    try:
        pred = modelo(img_array, verbose=False)[0]
    except Exception as e:
        raise RuntimeError(f"Error ejecutando la predicción del modelo: {e}.")

    if pred.masks is None:
        return []
    return pred.masks.data.cpu().numpy()


def combinar_predicciones(predicciones, shape):
    """Combina una lista de predicciones (máscaras binarias) en una sola máscara 2D."""
    height, width = shape
    prediccion_combinada = np.zeros((height, width), dtype=np.uint8)

    for pred in predicciones:
        binary = (pred > 0.5).astype(np.uint8)
        resized = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
        prediccion_combinada = np.maximum(prediccion_combinada, resized)

    return prediccion_combinada


def normalizar_prediccion(pred):
    """Normaliza orientación y escala de la predicción (0-255, formato axial)."""
    pred_normalizada = cv2.flip(pred.T, 1)
    pred_normalizada *= 255
    return pred_normalizada


def guardar_prediccion(pred, image_filename, output_dir):
    """Guarda la predicción binaria en formato PNG."""
    if pred is None or pred.size == 0:
        logger.warning(f"⚠️ Predicción vacía para {image_filename}, no se guardó nada.")
        return None

    crear_directorio(output_dir)
    output_path = output_dir / f"{image_filename}.png"

    if not ruta_existente(output_path):
        cv2.imwrite(output_path, pred, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return output_path


def predicciones_fold_completas(fold_dir, plano):
    """
    Devuelve True si todos los pacientes en fold_dir tienen pred_masks no vacíos.
    """
    for paciente_id in listar_pacientes(fold_dir):
        paciente_pred_masks_dir = fold_dir / paciente_id / plano / "pred_masks"
        if not paciente_pred_masks_dir.exists() or not any(
            paciente_pred_masks_dir.glob("*.png")
        ):
            return False
    return True


# =============================
#     PREDICCIÓN POR IMAGEN
# =============================


def generar_prediccion_2D(modelo, img_array, image_filename, output_dir):
    """Aplica el modelo YOLO a una imagen y guarda la predicción binaria resultante."""
    lista_predicciones_brutas = ejecutar_prediccion(modelo=modelo, img_array=img_array)
    prediccion_combinada = combinar_predicciones(
        predicciones=lista_predicciones_brutas, shape=img_array.shape[:2]
    )
    prediccion_normalizada = normalizar_prediccion(pred=prediccion_combinada)
    guardar_prediccion(
        pred=prediccion_normalizada,
        image_filename=image_filename,
        output_dir=output_dir,
    )


def obtener_imagenes_paciente(paciente_id, imagenes_dir):
    """Devuelve una lista de rutas completas a las imágenes PNG del paciente."""
    if not ruta_existente(imagenes_dir):
        raise FileNotFoundError(f"El directorio {imagenes_dir} no existe.")

    imagenes = sorted(
        [img for img in imagenes_dir.glob(f"{paciente_id}_*.png") if img.is_file()]
    )

    if not imagenes:
        raise FileNotFoundError(
            f"No se encontraron imágenes PNG para {paciente_id} en {imagenes_dir}."
        )
    return imagenes


def generar_predicciones(modelo, lista_imagenes, output_dir):
    """Aplica un modelo YOLO sobre todas las imágenes de un paciente."""
    for image_path in lista_imagenes:
        image_filename = image_path.stem
        img_array = cv2.imread(str(image_path))

        if img_array is None:
            logger.warning(f"⚠️ No se pudo cargar la imagen {image_path}.")
            continue

        generar_prediccion_2D(
            modelo=modelo,
            img_array=img_array,
            image_filename=image_filename,
            output_dir=output_dir,
        )


# =============================
#        PROCESAMIENTO
# =============================


def procesar_paciente_predicciones(
    paciente_id, config, paths_dir=None, modelo_cargado=None
):
    """
    Ejecuta el proceso completo de predicción
    (carga de modelo → predicciones → guardado de máscaras)
    para un paciente individual.
    """
    # Si no hay modelo pasado → cargarlo
    if modelo_cargado is None:
        modelo_cargado = cargar_modelo(config.model_path)

    # Si no se pasan los directorios → modo paciente → asigna los del config
    if paths_dir is None:
        paths_dir = config.paciente_dir

    images_paciente = paths_dir["images"]
    pred_masks_paciente = paths_dir["pred_masks"]

    # Evitar reprocesar si ya existen resultados
    if ruta_existente(pred_masks_paciente) and any(pred_masks_paciente.glob("*.png")):
        return

    # Obtener imágenes del paciente
    lista_imagenes = obtener_imagenes_paciente(
        paciente_id=paciente_id, imagenes_dir=images_paciente
    )

    if lista_imagenes == 0:
        raise RuntimeError(f"No hay imágenes válidas en {images_paciente}.")

    generar_predicciones(
        modelo=modelo_cargado,
        lista_imagenes=lista_imagenes,
        output_dir=pred_masks_paciente,
    )
    return True


def construir_paths(paciente_id, config):
    """
    Construye un diccionario de paths (images, pred_masks)
    para un paciente individual.
    """
    root = config.dataset_fold_dir / paciente_id / config.plano
    return {
        "images": root / "images",
        "pred_masks": root / "pred_masks",
    }


def generar_prediccion_por_paciente(input_dir, config):
    """
    Ejecuta el proceso de generación de predicciones para todos los pacientes en input_dir.
    """
    pacientes = listar_pacientes(input_dir)
    modelo = cargar_modelo(config.model_path)
    resultados = []

    for paciente_id in tqdm(pacientes, desc=f"Pacientes {input_dir.name}", unit="pac"):
        paths_paciente = construir_paths(paciente_id, config)
        try:
            pred_paciente = procesar_paciente_predicciones(
                paciente_id=paciente_id,
                config=config,
                paths_dir=paths_paciente,
                modelo_cargado=modelo,
            )
            resultados.append(pred_paciente)
        except Exception as e:
            logger.warning(
                f"⚠️ Error generando predicciones de {paciente_id}, se omite: {e}."
            )
            continue

    return evaluar_resultados(resultados)


# =============================
#        FLUJO PRINCIPAL
# =============================


def ejecutar_flujo_pred(config, limpiar, verbose=False):
    """Ejecuta el flujo principal de predicción, ya sea sobre un fold o un paciente individual."""
    if verbose:
        str_fold = f"fold {config.fold_test}"
        str_paciente = f"paciente {config.paciente}"
        logger.erer(
            f"\n🎯 Generando predicciones para el {str_paciente if config.es_paciente_individual else str_fold}."
        )

    # Limpiar si corresponde
    if limpiar:
        if verbose:
            logger.info(f"♻️ Limpiando predicciones previas.")
        config.limpiar_predicciones()

    # Verificar paths
    config.verificar_paths()

    # Ejecución por paciente
    if config.es_paciente_individual:
        predicciones_generadas = procesar_paciente_predicciones(
            paciente_id=config.paciente.id, config=config
        )
        if predicciones_generadas is None:
            logger.skip(f"⏩ Predicciones ya existente.")
        elif predicciones_generadas is True:
            logger.info(f"✅ Predicciones generadas correctamente.")
        else:
            logger.warning(f"⚠️ Estado desconocido al generar predicciones.")

    # Ejecución por fold
    else:
        # Chequear si el fold completo ya tiene predicciones
        if predicciones_fold_completas(config.dataset_fold_dir, config.plano):
            logger.skip(f"⏩ Fold {config.fold_test} ya existente.")
            return

        procesados = generar_prediccion_por_paciente(
            input_dir=config.dataset_fold_dir, config=config
        )
        log_estado_fold(logger=logger, resultado=procesados, fold=config.fold_test)


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
        help="Generar las predicciones 2D para el fold indicado, utilizado como conjunto de test.",
    )
    group.add_argument(
        "--paciente_id",
        type=str,
        metavar="<paciente_id>",
        help="Generar las predicciones 2D solo para el paciente indicado.",
    )
    parser.add_argument(
        "--limpiar",
        action="store_true",
        default=False,
        help="Limpiar las predicciones 2D binarias generadas previamente.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    """
    Entrada CLI del script: parsea argumentos, construye Modelo/Paciente/ConfigPred
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
        config = ConfigPred(
            modelo=modelo,
            epochs=args.epochs,
            k_folds=args.k_folds,
            paciente=paciente,
        )

    # Ejecución por fold
    else:
        config = ConfigPred(
            modelo=modelo,
            epochs=args.epochs,
            k_folds=args.k_folds,
            fold_test=args.fold_test,
        )

    ejecutar_flujo_pred(config=config, limpiar=args.limpiar, verbose=True)


def ejecutar_predicciones_pipeline(
    modelo, paciente=None, fold_test=None, epochs=50, k_folds=5, limpiar=False
):
    """
    Entrada interna para el pipeline: recibe objetos ya
    construidos y ejecuta el flujo sin usar el parser CLI.
    """
    # Ejecución por paciente
    if paciente is not None:
        config = ConfigPred(
            modelo=modelo,
            epochs=epochs,
            k_folds=k_folds,
            paciente=paciente,
        )

    # Ejecución por fold
    elif fold_test is not None:
        config = ConfigPred(
            modelo=modelo,
            epochs=epochs,
            k_folds=k_folds,
            fold_test=fold_test,
        )

    else:
        raise ValueError("Debe especificarse un paciente o un fold de test.")

    ejecutar_flujo_pred(
        config=config,
        limpiar=limpiar,
    )


if __name__ == "__main__":
    main()
