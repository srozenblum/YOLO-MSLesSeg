"""
Script: setup.py

Descripción:
    Descarga automáticamente el dataset MSLesSeg desde el repositorio oficial
    (Figshare), descomprime el archivo ZIP removiendo posibles carpetas
    intermedias y organiza la estructura base del dataset en el directorio
    MSLesSeg-Dataset/. Además, genera el directorio GT/ con las máscaras
    ground truth de cada paciente, de acuerdo a lo requerido para la ejecución
    del pipeline.

Modos de ejecución:
    1. CLI (uso independiente).
    2. Interno (desde `ejecutar_pipeline.py`).

Argumentos CLI:
    --url (str, opcional)
        Enlace directo de descarga al archivo ZIP del dataset MSLesSeg.
        Por defecto apunta al archivo oficial en Figshare.

    --limpiar (flag, opcional)
        Limpiar el directorio GT/ generado previamente, pero no el
        dataset descargado en MSLesSeg-Dataset/.

Uso por CLI:
        python -m yolo_mslesseg.scripts.setup --limpiar

Entradas:
    - URL al archivo ZIP del dataset MSLesSeg
      (https://springernature.figshare.com/ndownloader/files/52771814).

Salidas:
    - Directorio MSLesSeg-Dataset/ con la estructura limpia del dataset oficial
      sin carpetas intermedias.

    - Directorio GT/ con las máscaras ground truth:
        GT/train/PX/PX_MASK.nii.gz
        GT/test/PX/PX_MASK.nii.gz
"""

import argparse
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from yolo_mslesseg.utils.configurar_logging import get_logger
from yolo_mslesseg.utils.utils import crear_directorio, eliminar_directorio

# Configurar logger
logger = get_logger(__file__)


# =============================
#     FUNCIONES AUXILIARES
# =============================


def dataset_existente(dataset_root):
    """
    Verifica si el dataset MSLesSeg ya existe
    (es decir, si existen las carpetas train/ o test/).
    """
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    return train_dir.exists() or test_dir.exists()


def gt_existente(gt_root):
    """
    Verifica si el directorio de ground truth ya existe
    (es decir, si existen las carpetas train/ y test/).
    """
    gt_train = gt_root / "train"
    gt_test = gt_root / "test"
    return gt_train.exists() and gt_test.exists()


# =============================
#           DESCARGA
# =============================


def descargar_archivo(url, destino):
    respuesta = requests.get(url, stream=True)
    respuesta.raise_for_status()

    tamanio_total = int(respuesta.headers.get("content-length", 0))
    chunk = 1024 * 1024  # 1 MB

    with open(destino, "wb") as f, tqdm(
        total=tamanio_total, unit="B", unit_scale=True, desc=f"{destino.name}", ncols=80
    ) as barra:
        for bloque in respuesta.iter_content(chunk_size=chunk):
            if bloque:
                f.write(bloque)
                barra.update(len(bloque))


# =============================
#        DESCOMPRESIÓN
# =============================


def descomprimir_zip(archivo_zip, destino):

    with zipfile.ZipFile(archivo_zip, "r") as zip_ref:
        nombres = zip_ref.namelist()

        # Detectar carpeta raíz común (ej. "MSLesSeg_dataset/")
        carpeta_raiz = None
        primeros = [n.split("/")[0] for n in nombres if "/" in n]

        if len(set(primeros)) == 1:
            carpeta_raiz = list(set(primeros))[0] + "/"

        for nombre in nombres:

            # Ignorar carpeta info_dataset y todo su contenido
            if "info_dataset/" in nombre:
                continue

            # Quitar carpeta raíz si existe
            nuevo_nombre = nombre
            if carpeta_raiz and nombre.startswith(carpeta_raiz):
                nuevo_nombre = nombre[len(carpeta_raiz) :]

            # Ignorar entradas vacías
            if not nuevo_nombre.strip():
                continue

            destino_final = destino / nuevo_nombre

            # Si es carpeta → crear
            if nombre.endswith("/"):
                destino_final.mkdir(parents=True, exist_ok=True)
                continue

            # Si es archivo → copiarlo
            destino_final.parent.mkdir(parents=True, exist_ok=True)
            with zip_ref.open(nombre) as src, open(destino_final, "wb") as dst:
                shutil.copyfileobj(src, dst)

            # Si el ZIP tiene carpeta intermedia, recortar esa parte
            if carpeta_raiz and nombre.startswith(carpeta_raiz):
                nuevo_nombre = nombre[len(carpeta_raiz) :]

            # Ignorar entradas vacías (directorios)
            if not nuevo_nombre.strip():
                continue

            destino_final = destino / nuevo_nombre

            # Si es carpeta, crearla
            if nombre.endswith("/"):
                destino_final.mkdir(parents=True, exist_ok=True)
            else:
                destino_final.parent.mkdir(parents=True, exist_ok=True)
                with zip_ref.open(nombre) as src, open(destino_final, "wb") as dst:
                    shutil.copyfileobj(src, dst)


# ===============================
#  ORGANIZACIÓN DE DIRECTORIO GT
# ===============================


def obtener_mask_path(paciente_dir, split):
    """
    Devuelve la ruta de la máscara correspondiente al paciente según el split.
    - train → PX/T1/PX_T1_MASK.nii.gz
    - test  → PX/PX_MASK.nii.gz
    """
    paciente_id = paciente_dir.name

    if split == "train":
        return paciente_dir / "T1" / f"{paciente_id}_T1_MASK.nii.gz"
    else:  # test
        return paciente_dir / f"{paciente_id}_MASK.nii.gz"


def copiar_mask(mask_path, gt_root, split, paciente_id):
    """
    Copia la máscara ground truth al directorio GT/,
    unificando los nombres a PX_MASK.nii.gz.
    """
    destino_gt_paciente = gt_root / split / paciente_id
    crear_directorio(destino_gt_paciente)

    nuevo_nombre = f"{paciente_id}_MASK.nii.gz"
    shutil.copy2(mask_path, destino_gt_paciente / nuevo_nombre)


def procesar_split(dataset_root, gt_root, split):
    """
    Recorre los pacientes de un split (train/test), localiza sus máscaras y las copia.
    """
    split_root = dataset_root / split
    if not split_root.exists():
        return

    for paciente_dir in sorted(split_root.iterdir()):
        if not paciente_dir.is_dir():
            continue

        paciente_id = paciente_dir.name

        mask_path = obtener_mask_path(paciente_dir, split)
        if not mask_path.exists():
            continue

        copiar_mask(mask_path, gt_root, split, paciente_id)


def mover_volumenes_gt(dataset_root, gt_root):
    """
    Genera la estructura GT/train/ y GT_/test/, copiando
    las máscaras originales del dataset con nombres de
    archivo unificados.
    """
    crear_directorio(gt_root)
    crear_directorio(gt_root / "train")
    crear_directorio(gt_root / "test")

    procesar_split(dataset_root, gt_root, "train")
    procesar_split(dataset_root, gt_root, "test")


# =============================
#        PROCESAMIENTO
# =============================


def procesar_descarga_y_descompresion(dataset_root, url):
    """
    Ejecuta el proceso de descarga y descompresión
    del dataset de entrada MSLesSeg.
    """
    crear_directorio(dataset_root)
    zip_path = dataset_root / "MSLesSeg_dataset.zip"

    # Descargar el archivo comprimido desde la URL
    try:
        descargar_archivo(url=url, destino=zip_path)
    except:
        raise

    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Archivo ZIP inválido.")

    # Descomprimir y eliminar el ZIP
    logger.info(f"🗂️ Descomprimiendo {zip_path}...")
    try:
        descomprimir_zip(zip_path, dataset_root)
        zip_path.unlink()
    except:
        raise

    logger.info(
        f"✅ Descarga y descompresión completada correctamente en {dataset_root}/."
    )


def procesar_directorio_gt(dataset_root, gt_root):
    """
    Ejecuta el proceso de construcción del directorio GT/,
    copiando y unificando las máscaras originales del dataset
    en la estructura final requerida.
    """
    logger.info(f"📂 Generando directorio de ground truth (GT/)...")
    try:
        mover_volumenes_gt(dataset_root=dataset_root, gt_root=gt_root)
        logger.info(f"✅ Directorio GT/ generado correctamente.")
    except:
        raise


# =============================
#        FLUJO PRINCIPAL
# =============================


def ejecutar_flujo(url, limpiar, verbose=False):
    """
    Ejecuta el flujo principal de setup.
    """
    if verbose:
        logger.header(f"📦 Descargando dataset MSLesSeg")

    dataset_root = Path("MSLesSeg-Dataset")
    gt_root = Path("GT")

    # Limpiar solo GT (no borrar el dataset descargado)
    if limpiar:
        if verbose:
            logger.info(f"♻️ Limpiando directorio GT/ previo.")
        eliminar_directorio(gt_root)

    # Estado tras la limpieza
    dataset_existe = dataset_existente(dataset_root)
    gt_existe = gt_existente(gt_root)

    # 1) Dataset y GT existentes → skip
    if dataset_existe and gt_existe:
        logger.skip("⏩ Dataset de entrada y directorio de ground truth ya existentes.")
        return

    # 2) Dataset: descargar o reutilizar
    if dataset_existe:
        logger.info("⏩ Dataset de entrada ya existente.")
    else:
        procesar_descarga_y_descompresion(dataset_root=dataset_root, url=url)

    # 3) GT: generar o reutilizar
    if gt_existe:
        logger.info("⏩ Directorio GT/ ya existente.")
    else:
        procesar_directorio_gt(dataset_root=dataset_root, gt_root=gt_root)


# =============================
#        CLI Y EJECUCIÓN
# =============================


def parsear_args():
    """
    Parsea los argumentos del script
    leyendolos desde la línea de comandos (CLI).
    """

    parser = argparse.ArgumentParser(
        description="Descarga el dataset MSLesSeg desde Figshare y organiza la estructura de directorios para la ejecución del pipeline.",
    )
    parser.add_argument(
        "--limpiar",
        action="store_true",
        default=False,
        help="Limpiar solo el directorio GT/ generado previamente, pero no MSLesSeg-Dataset/.",
    )
    return parser.parse_args()


def main():
    """
    Entrada CLI del script:
    parsea argumentos y ejecuta el flujo completo.
    """
    args = parsear_args()

    ejecutar_flujo(
        url="https://springernature.figshare.com/ndownloader/files/52771814",
        limpiar=args.limpiar,
        verbose=True,
    )


def ejecutar_setup_pipeline(limpiar=False):
    """
    Entrada interna para el pipeline:
    ejecuta el flujo sin usar el parser CLI.
    """
    ejecutar_flujo(
        url="https://springernature.figshare.com/ndownloader/files/52771814",
        limpiar=limpiar,
    )


if __name__ == "__main__":
    main()
