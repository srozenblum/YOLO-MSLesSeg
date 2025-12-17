"""
Script: train.py

Descripción:
    Ejecuta el proceso de entrenamiento de un modelo YOLO sobre un fold
    en un esquema de validación cruzada. Puede ejecutarse tanto de forma
    independiente (desde CLI) como internamente dentro del pipeline
    (`ejecutar_pipeline.py`).

    Genera los conjuntos de entrenamiento y prueba en formato YOLO, crea
    el archivo YAML correspondiente, y entrena el modelo durante el número
    de épocas indicado. Este script asume que el conjunto de datos ya ha
    sido generado por `extraer_dataset.py` y que se encuentra dividido en
    folds dentro del directorio datasets/.

Modos de ejecución:
    1. CLI (uso independiente):
       - Se leen y parsean los argumentos escritos por el usuario en la línea de comandos.
       - Se crea la instancia de Modelo.

    2. Interno (desde `ejecutar_pipeline.py`):
       - Se recibe instancia ya creada de Modelo, junto con el resto de parámetros.
       - No se usa el parser de argumentos.

Argumentos CLI:
   --plano (str, requerido)
        Plano anatómico del modelo ('axial', 'coronal', 'sagital').

   --modalidad (list[str], opcional)
        Modalidad o modalidades de imagen ('T1', 'T2', 'FLAIR'). Por defecto todas.

   --num_cortes (int_o_percentil, requerido)
        Número de cortes a extraer (valor entero o percentil, por ejemplo 50 o 'P75').

   --mejora (str, opcional)
        Algoritmo de mejora de imagen ('HE', 'CLAHE', 'GC', 'LT'). Por defecto None.

   --epochs (int, requerido)
        Número de épocas de entrenamiento.

   --k_folds (int, opcional)
        Número de folds para validación cruzada. Por defecto 5.

   --fold_test (int, requerido)
        Fold utilizado como conjunto de test (1, ..., k_folds).

   --limpiar(flag, opcional)
        Limpiar los resultados previos de entrenamiento antes de iniciar una nueva ejecución.

Uso por CLI:
    python -m yolo_mslesseg.scripts.train \
        --plano "sagital" \
        --modalidad "T2" \
        --num_cortes 20 \
        --epochs 40 \
        --fold_test 2 \

Entradas:
    - Dataset: generado previamente con `extraer_dataset.py`.
        Contiene las imágenes, máscaras GT y etiquetas de cada paciente, divididos en folds.

    - Clases:
        * ConfigTrain → gestiona directorios y variables globales asociados al entrenamiento.
        * Modelo → define el plano, modalidades, mejora y número de cortes del experimento.

Salidas:
    - Resultados del entrenamiento YOLO en el directorio trains/.
    - Archivo YAML de configuración del experimento por fold.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import yaml
from ultralytics.utils import LOGGER

from yolo_mslesseg.configs.ConfigTrain import ConfigTrain
from yolo_mslesseg.utils.Modelo import Modelo
from yolo_mslesseg.utils.configurar_logging import get_logger
from yolo_mslesseg.utils.utils import (
    int_o_percentil,
    eliminar_directorio,
    cargar_modelo,
    archivo_ignorable,
    crear_directorio,
    ruta_existente,
    existe_modelo_entrenado,
)

# Configurar logger
logger = get_logger(__file__)


# Ocultar loggings de ultralytics
LOGGER.setLevel(logging.ERROR)

# =============================
#     FUNCIONES AUXILIARES
# =============================


def entrenamiento_exitoso(fold_train_dir):
    """
    Comprueba si el entrenamiento fue exitoso a partir de
    los archivos esenciales del entrenamiento YOLO:
        - weights/best.pt   → mejores pesos obtenidos
        - weights/last.pt   → últimos pesos obtenidos
        - results.csv       → resumen de métricas del entrenamiento
    """
    best = fold_train_dir / "weights" / "best.pt"
    last = fold_train_dir / "weights" / "last.pt"
    results = fold_train_dir / "results.csv"
    return best.is_file() and last.is_file() and results.is_file()


# =============================
#     MANEJO DE DIRECTORIOS
# =============================


def obtener_folds_existentes(dataset_dir):
    """Devuelve la lista de folds existentes en el dataset."""
    folds = []
    for d in dataset_dir.iterdir():
        if d.is_dir() and d.name.startswith("fold"):
            try:
                n = int(d.name.replace("fold", ""))
                folds.append(n)
            except ValueError:
                pass  # No es un fold válido → se ignora

    return sorted(folds)


def copiar_contenido_directorio(input_dir, output_dir):
    """
    Copia el contenido de input_dir a output_dir,
    omitiendo archivos ocultos o de sistema.
    Input_dir no se modifica luego de la ejecución.
    """
    for f in input_dir.iterdir():
        if archivo_ignorable(f.name):
            continue
        dst = output_dir / f.name
        try:
            if f.is_dir():  # Si es un directorio → copiarlo de forma recursiva
                shutil.copytree(f, dst, dirs_exist_ok=True)
            else:  # Si es un archivo → simplemente copiarlo
                shutil.copy2(f, dst)
        except Exception as e:
            logger.error(f"❌ Error copiando '{f}' → '{dst}': {e}")
            raise


# =============================
#   PROCESAMIENTO DE FOLDS
# =============================


def copiar_pacientes_de_fold(fold_dir, plano, output_dir):
    """
    Copia todas las imágenes y etiquetas correspondientes al plano dado
    desde un directorio de fold hacia un directorio plano de salida.

    Estructura de entrada esperada:
        fold_dir/PX/<plano>/images/*.png
        fold_dir/PX/<plano>/labels/*.txt

    Estructura de salida:
        output_dir/  (todos los .png y .txt juntos)
    """
    for paciente_dir in fold_dir.iterdir():
        if not paciente_dir.is_dir():
            continue

        for plano_dir in paciente_dir.iterdir():
            if not (plano_dir.is_dir() and plano_dir.name.startswith(plano)):
                continue

            images_dir = plano_dir / "images"
            labels_dir = plano_dir / "labels"

            copiar_contenido_directorio(images_dir, output_dir)
            copiar_contenido_directorio(labels_dir, output_dir)


def duplicar_labels_modalidades(images_dir, labels_dir):
    """
    Para cada imagen PX_<modalidad>_<corte>.png, crea una
    etiqueta PX_<modalidad>_<corte>.txt copiando el contenido
    de PX_<corte>.txt, la cual es luego eliminada.
    """
    # Agrupar todas las etiquetas base detectadas para borrarlas al final
    labels_base = set()

    for img in images_dir.glob("*.png"):
        parts = img.stem.split("_")  # ['P8', 'T1', '102']
        if len(parts) != 3:
            continue

        paciente, mod, corte = parts
        label_base = labels_dir / f"{paciente}_{corte}.txt"
        label_dest = labels_dir / f"{paciente}_{mod}_{corte}.txt"

        if label_base.exists():
            if not label_dest.exists():
                shutil.copy2(label_base, label_dest)
            labels_base.add(label_base)

    # Eliminar todas las etiquetas base
    for lb in labels_base:
        try:
            lb.unlink()
        except Exception as e:
            logger.warning(f"⚠️ No se pudo eliminar {lb}: {e}.")


def preparar_fold_yolo(config, tipo="train"):
    """
    Organiza el contenido de un fold en el formato requerido por YOLO.
    Mueve todas las imágenes (.png) a 'images/' y todas las etiquetas (.txt)
    a 'labels/' dentro del fold correspondiente.
    """
    if tipo == "train":
        fold_dir = config.fold_train_dir
    elif tipo == "test":
        fold_dir = config.fold_test_dir
    else:
        raise ValueError("Debe especificarse tipo 'train' o 'test'.")

    images_output_dir = fold_dir / "images"
    labels_output_dir = fold_dir / "labels"
    crear_directorio(images_output_dir)
    crear_directorio(labels_output_dir)

    # Mover imágenes .png → images/
    for img in fold_dir.glob("*.png"):
        shutil.move(str(img), str(images_output_dir / img.name))

    # Mover etiquetas .txt → labels/
    for lbl in fold_dir.glob("*.txt"):
        shutil.move(str(lbl), str(labels_output_dir / lbl.name))

    # Estandarizar anotaciones
    duplicar_labels_modalidades(images_output_dir, labels_output_dir)


# =============================
#   GENERACIÓN DE SUBCONJUNTOS
# =============================


def crear_train_subset(config):
    """Construye el conjunto de entrenamiento para el fold actual combinando los demás folds."""
    fold_actual = config.fold_test
    folds_existentes = obtener_folds_existentes(config.dataset_base_dir)
    train_output_dir = config.fold_train_dir  # Ruta de salida

    # Folds a combinar: todos excepto el actual
    fold_input_dirs = [
        config.dataset_base_dir / f"fold{i}"
        for i in folds_existentes
        if i != fold_actual
    ]

    # Eliminar train anterior si existe (para evitar mezclas de contenido) y volver a crear
    if ruta_existente(train_output_dir):
        eliminar_directorio(train_output_dir)
    crear_directorio(train_output_dir)

    # Combinar todos los folds seleccionados
    for fold_dir in fold_input_dirs:
        if not ruta_existente(fold_dir):
            logger.warning(f"⚠️ Fold no encontrado: {fold_dir}. Se omite.")
            continue

        try:
            copiar_pacientes_de_fold(
                fold_dir=fold_dir, plano=config.plano, output_dir=train_output_dir
            )
        except Exception as e:
            logger.error(f"❌ Error al copiar {fold_dir} hacia train: {e}")
            raise

    # Reorganizar estructura del fold al formato YOLO
    preparar_fold_yolo(config, tipo="train")


def crear_test_subset(config):
    """Crea el conjunto de test a partir del fold actual."""
    # Rutas de entrada y salida
    fold_input_dir = config.fold_dir
    test_output_dir = config.fold_test_dir

    # Eliminar test anterior si existe (para evitar mezclas de contenido) y volver a crear
    if ruta_existente(test_output_dir):
        eliminar_directorio(test_output_dir)
    crear_directorio(test_output_dir)

    # Copiar el contenido completo del fold al directorio de test
    copiar_pacientes_de_fold(
        fold_dir=fold_input_dir, plano=config.plano, output_dir=test_output_dir
    )

    # Reorganizar estructura del fold al formato YOLO
    preparar_fold_yolo(config, tipo="test")


# =============================
#    CONFIGURACIÓN DE YOLO
# =============================


def generar_yaml(config):
    """Genera un diccionario de configuración YOLO."""
    return {
        "path": str(config.dataset_base_dir.parent.resolve()),
        "train": str(config.fold_train_dir.resolve()),
        "val": str(config.fold_test_dir.resolve()),
        "names": ["lesion"],
        "nc": 1,
    }


def guardar_yaml(yolo_dict, config):
    """Guarda el diccionario de configuración YOLO en el archivo YAML."""
    with open(config.yaml_path, "w") as f:
        yaml.dump(yolo_dict, f, default_flow_style=False, sort_keys=False)


def cargar_yaml(config):
    """Carga el archivo YAML asociado a la configutación y lo devuelve como diccionario."""
    with open(config.yaml_path, "r") as f:
        yolo_dict = yaml.safe_load(f)
    return yolo_dict


# =============================
#        ENTRENAMIENTO
# =============================


def entrenar_fold(config):
    """Ejecuta el entrenamiento de un fold para un modelo."""
    # Crear conjuntos de train y test
    crear_train_subset(config)
    crear_test_subset(config)

    # Generar archivo YAML de configuración para YOLO
    yolo_dict = generar_yaml(config)
    guardar_yaml(yolo_dict, config)

    # Ejecutar entrenamiento
    model_yolo = cargar_modelo(config.weights_path)
    model_yolo.train(
        data=config.yaml_path,
        epochs=config.epochs,
        batch=-1,
        cache=True,
        project=config.train_output_dir,
        name=f"fold{config.fold_test}",
        verbose=False,
    )

    # Copiar YAML al directorio del experimento
    yaml_dest = (
        config.train_output_dir
        / f"fold{config.fold_test}"
        / f"{config.modelo.model_string}.yaml"
    )
    try:
        shutil.copy2(config.yaml_path, yaml_dest)
    except Exception as e:
        logger.warning(
            f"⚠️ No se pudo copiar el YAML al directorio del experimento: {e}."
        )

    # Eliminar conjuntos de train y test
    eliminar_directorio(Path(config.fold_train_dir).parent)
    eliminar_directorio(Path(config.fold_test_dir).parent)


# =============================
#       FLUJO PRINCIPAL
# =============================


def ejecutar_flujo_train(config, limpiar, verbose=False):
    """Ejecuta el flujo principal de entrenamiento de un fold."""
    if verbose:
        logger.header(f"🧠️ Entrenando modelo (test = fold{config.fold_test})")

    if limpiar:
        config.limpiar_entrenamiento()
        if verbose:
            logger.info(f"♻️ Limpiando entrenamiento previo.")

    config.verificar_paths()

    if existe_modelo_entrenado(
        modelo=config.modelo, epochs=config.epochs, fold_test=config.fold_test
    ):
        logger.skip(f"⏩ Entrenamiento ya existente para fold {config.fold_test}.")

    else:
        entrenar_fold(config)

        fold_dir = config.train_output_dir / f"fold{config.fold_test}"

        if entrenamiento_exitoso(fold_dir):
            logger.info("✅ Entrenamiento completado correctamente.")
        else:
            logger.warning(
                "⚠️ El entrenamiento no devolvió resultados. Puede haberse interrumpido."
            )


# =============================
#        CLI Y EJECUCIÓN
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
        description="Ejecutar el entrenamiento de un fold de un modelo YOLO.",
    )
    parser.add_argument(
        "--plano",
        type=str,
        required=True,
        choices=["axial", "coronal", "sagital"],
        metavar="[axial, coronal, sagital]",
        help="Plano anatómico del modelo [axial, coronal, sagital].",
    )
    parser.add_argument(
        "--modalidad",
        nargs="+",
        choices=["T1", "T2", "FLAIR"],
        default=["T1", "T2", "FLAIR"],
        metavar="[T1, T2, FLAIR]",
        help="Modalidad(es) de imagen MRI [T1, T2, FLAIR]. Por defecto todas.",
    )
    parser.add_argument(
        "--num_cortes",
        type=int_o_percentil,
        required=True,
        metavar="<num_cortes>",
        help="Número de cortes extraídos [valor fijo o percentil].",
    )
    parser.add_argument(
        "--mejora",
        type=str,
        default=None,
        choices=["HE", "CLAHE", "GC", "LT"],
        metavar="[HE, CLAHE, GC, LT]",
        help="Algoritmo de mejora de imagen aplicado [HE, CLAHE, GC, LT]. Por defecto None.",
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
        "--fold_test",
        type=int,
        required=True,
        metavar="<fold_test>",
        help="Fold utilizado como conjunto de test (1, ..., k_folds).",
    )
    parser.add_argument(
        "--limpiar",
        action="store_true",
        help="Limpiar los resultados del entrenamiento generados previamente.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    """
    Entrada CLI del script: parsea argumentos, construye Modelo/Paciente/ConfigTrain
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

    config = ConfigTrain(
        modelo=modelo,
        fold_test=args.fold_test,
        epochs=args.epochs,
    )

    ejecutar_flujo_train(config=config, limpiar=args.limpiar, verbose=True)


def ejecutar_train_pipeline(modelo, fold_test, epochs=50, limpiar=False):
    """
    Entrada interna para el pipeline: recibe objetos ya
    construidos y ejecuta el flujo sin usar el parser CLI.
    """
    config = ConfigTrain(
        modelo,
        fold_test=fold_test,
        epochs=epochs,
    )
    ejecutar_flujo_train(config=config, limpiar=limpiar)


if __name__ == "__main__":
    main()
