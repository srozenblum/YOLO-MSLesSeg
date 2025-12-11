"""
Script: utils.py

Descripción:
    Conjunto de utilidades comunes utilizadas como apoyo para los scripts
    principales, proporcionando funciones reutilizables y homogéneas que
    evitan duplicar lógica entre diferentes etapas.

Bloques funcionales principales:
    - Directorios y archivos:
        Creación, eliminación y validación de rutas; filtrado de archivos
        irrelevantes.

    - Volúmenes NIfTI:
        Carga, guardado, validación dimensional y reconstrucción de volúmenes 3D.

    - Modelos YOLO:
        Carga segura de modelos y manejo básico de errores.

    - JSON:
        Lectura y escritura de diccionarios en formato JSON.

    - Pacientes y folds:
        Obtención de IDs, listado, y división en folds.

    - Cálculo de percentiles:
        Tipo de dato personalizado int_o_percentil y cálculo de percentil.

    - Procesamiento de imagen:
        Normalización de máscaras binarias, conversión a uint8, conversión RGB/BGR
        y normalización a escala de grises.

    - Métricas y evaluación:
        Cálculo de métricas de rendimiento y evaluación de resultados parciales.

    - Logging de estado:
        Funciones auxiliares para registrar el estado de ejecución de un fold
        dentro de etapas del pipeline (p. ej. predicción, reconstrucción o evaluación),
        incluyendo compatibilidad con niveles personalizados como SKIP.

Modos de uso:
    - Interno: se importa desde cualquier etapa del pipeline.
    - No está diseñado para ejecución directa por CLI.

Convenciones:
    - Todas las rutas se manejan con pathlib.Path.
    - Las funciones nunca silencian errores críticos: relanzan excepciones.
    - Las máscaras binarias se normalizan al rango {0, 1}.
    - El tipo `int_o_percentil` admite valores enteros o strings "P<n>".
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from ultralytics import YOLO

from yolo_mslesseg.utils.configurar_logging import get_logger

# Configurar logger
logger = get_logger(__file__)


# ===============================
#     DIRECTORIOS Y ARCHIVOS
# ===============================


def ruta_existente(path):
    """Verifica que la ruta exista."""
    return Path(path).exists()


def crear_directorio(path):
    """Crea el directorio si no existe."""
    Path(path).mkdir(parents=True, exist_ok=True)


def eliminar_directorio(input_dir):
    """Elimina el directorio de forma recursiva."""
    path = Path(input_dir)
    if path.exists() and path.is_dir():
        shutil.rmtree(path)


def archivo_ignorable(nombre):
    """Devuelve True si el archivo parece de sistema u oculto."""
    nombre_lower = nombre.lower()
    return (
        nombre.startswith(".")
        or nombre.startswith("~")
        or nombre_lower.endswith(".tmp")
    )


def construir_nombre_configuracion(modelo, epochs):
    """
    Construye el nombre de la carpeta de configuración global del modelo
    (modalidad, número de cortes, k_folds y epochs).
    """
    modalidades = "".join(modelo.modalidad)  # Ej: ['FLAIR'] → "FLAIR"
    return f"{modalidades}_{modelo.num_cortes}c_{modelo.k_folds}folds_{epochs}epochs"


def base_dir_paciente(paciente, modelo):
    """
    Devuelve el directorio base donde se almacenan las imágenes, predicciones
    y máscaras ground truth de un paciente dentro del dataset YOLO.
    """
    repo_root = Path(__file__).resolve().parents[2]

    paciente_id = paciente.id
    plano = paciente.plano

    k_folds = modelo.k_folds
    fold = calcular_fold(paciente_id, k_folds)

    return (
        repo_root / "datasets" / modelo.base_path / f"fold{fold}" / paciente_id / plano
    )


def paths_paciente(paciente, modelo, corte):
    """
    Construye y devuelve un diccionario con las rutas a la imagen, la predicción y
    la máscara ground truth de un corte específico del paciente.
    """
    paciente_id = paciente.id
    modalidad = paciente.modalidad_str

    base_dir = base_dir_paciente(paciente=paciente, modelo=modelo)

    return {
        "img": base_dir / "images" / f"{paciente_id}_{modalidad}_{corte}.png",
        "pred": base_dir / "pred_masks" / f"{paciente_id}_{modalidad}_{corte}.png",
        "gt": base_dir / "GT_masks" / f"{paciente_id}_{corte}.png",
    }


# ===============================
#   MANEJO DE VOLÚMENES NIFTI
# ===============================


def cargar_volumen(vol_path):
    """Carga un archivo NIfTI y devuelve su array de datos."""
    try:
        return nib.load(vol_path).get_fdata()
    except Exception as e:
        logger.error(f"❌ Error al cargar el volumen desde {vol_path}: {e}")
        raise


def cargar_referencia_nifti(referencia_path):
    """Carga un archivo NIfTI y devuelve su shape y affine."""
    if not ruta_existente(referencia_path):
        raise FileNotFoundError(f"Archivo no encontrado: {referencia_path}")
    try:
        nifti = nib.load(referencia_path)
        return nifti.shape, nifti.affine
    except nib.filebasedimages.ImageFileError as e:
        raise ValueError(f"Archivo no válido: {referencia_path}") from e


def guardar_volumen(volumen, affine, output_path):
    """Guarda un volumen NIfTI en la ruta de salida indicada."""
    try:
        nifti_out = nib.Nifti1Image(volumen, affine)
        nib.save(nifti_out, output_path)
    except Exception as e:
        logger.error(f"❌ Error al guardar el volumen en {output_path}: {e}")
        raise


def reconstruccion_valida(pred_vol_path, gt_vol_path):
    """
    Valida que la reconstrucción de un paciente sea consistente con su GT.
    """
    pred_vol = cargar_volumen(pred_vol_path)
    gt_vol = cargar_volumen(gt_vol_path)

    if pred_vol.shape != gt_vol.shape:
        logger.warning(f"⚠️ Dimensiones distintas: {pred_vol.shape} vs {gt_vol.shape}")
        return False

    return True


def volumenes_predichos_completos(paciente_dir):
    """
    Verifica que existan los tres volúmenes predichos (axial, coronal, sagital)
    para un paciente dentro de su directorio de predicciones.
    """
    planos = ["axial", "coronal", "sagital"]
    paciente_id = Path(paciente_dir).name
    return all(
        (Path(paciente_dir) / f"{paciente_id}_{plano}.nii.gz").exists()
        for plano in planos
    )


def verificar_volumenes_grupo(root_dir):
    """
    Verifica que todos los pacientes dentro de root_dir
    tengan los volúmenes predichos en los tres planos
    (axial, coronal y sagital).
    """
    pacientes = listar_pacientes(root_dir)
    pacientes_incompletos = []

    for paciente_id in pacientes:
        paciente_pred_root_dir = root_dir / paciente_id
        if not volumenes_predichos_completos(paciente_pred_root_dir):
            pacientes_incompletos.append(paciente_id)

    return pacientes_incompletos == []


# ===============================
#     MANEJO DE MODELOS YOLO
# ===============================


def cargar_modelo(model_path):
    """Carga el modelo YOLO."""
    try:
        return YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el modelo YOLO: {e}")


def existe_modelo_entrenado(modelo, epochs, fold_test):
    """Verifica si existen los pesos entrenados del modelo para un fold específico."""
    model_path = (
        Path("trains")
        / f"{modelo.base_path}_{epochs}epochs"
        / modelo.plano
        / f"fold{fold_test}"
        / "weights"
        / "best.pt"
    )

    return model_path.exists() and model_path.stat().st_size > 0


# ===============================
#             JSON
# ===============================


def escribir_json(dic, json_path):
    """Guarda un diccionario en formato JSON."""
    with open(json_path, "w") as f:
        json.dump(dic, f)


def leer_json(json_path):
    """Lee un archivo JSON y devuelve su contenido en un diccionario."""
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    raise FileNotFoundError(f"Archivo no encontrado: {json_path}")


# ===============================
# UTILIDADES DE PACIENTES Y FOLDS
# ===============================


def obtener_id(paciente):
    """Extrae el ID de un paciente desde su nombre (P12 → 12)."""
    match = re.search(r"P(\d+)", paciente)
    return (
        int(match.group(1)) if match else float("inf")
    )  # Devolver un valor muy grande si no se encuentra un número


def listar_pacientes(input_dir):
    """
    Devuelve una lista ordenada de IDs de pacientes en un directorio
    """
    input_path = Path(input_dir)

    pacientes = [d.name for d in input_path.iterdir() if not archivo_ignorable(d.name)]
    if not pacientes:
        raise FileNotFoundError(f"No se encontraron pacientes en {input_dir}.")

    return sorted(pacientes, key=lambda p: int(p[1:]) if p[1:].isdigit() else 1_000_000)


def calcular_fold(paciente_id, k_folds=5):
    """Asigna un paciente a su fold correspondiente (validación cruzada)."""

    # Convertir ID del paciente a número
    numero = int(paciente_id[1:])

    # Crear lista de IDs válidos (P1–P53)
    todos_los_ids = [i for i in range(1, 54)]

    # Dividir consecutivamente en k_folds
    folds = np.array_split(todos_los_ids, k_folds)

    # Buscar a qué fold pertenece el paciente
    for i, fold in enumerate(folds, 1):
        if numero in fold:
            return i

    raise ValueError(f"No se puede calcular el fold del paciente {paciente_id}.")


def obtener_cortes_paciente(paciente, modelo):
    """
    Devuelve una lista ordenada de los cortes disponibles para un paciente
    en un plano dado, extraídos a partir del subdirectorio images/ del dataset YOLO.
    """
    base_dir = base_dir_paciente(paciente=paciente, modelo=modelo)
    images_dir = base_dir / "images"

    cortes = []
    for fname in images_dir.glob("*.png"):
        try:
            corte = int(fname.stem.split("_")[-1])
            cortes.append(corte)
        except ValueError:
            continue  # Ignorar archivos que no sigan la convención

    return sorted(cortes)


# ===============================
#     MANEJO DE PERCENTILES
# ===============================


def int_o_percentil(valor):
    """Admite valores enteros o percentiles ('P<n>')."""
    try:
        return int(valor)
    except ValueError:
        if (
            isinstance(valor, str)
            and valor.upper().startswith("P")
            and valor[1:].isdigit()
        ):
            return valor.upper()
        raise argparse.ArgumentTypeError(
            "El valor debe ser un entero o un string de formato 'PX' (ejemplo: P10 para percentil 10)."
        )


def calcular_percentil(json_path, p, verbose=False):
    """Calcula el percentil deseado en función de la distribución de cortes"""
    cortes_dict = leer_json(json_path)
    cortes_por_paciente = [len(indices) for indices in cortes_dict.values()]

    if not cortes_por_paciente:
        raise ValueError("El archivo JSON no contiene pacientes o está vacío.")

    percentil_valor = int(np.percentile(cortes_por_paciente, p))
    if verbose:
        logger.info(
            f"📊 Se recomienda extraer {percentil_valor} cortes por paciente (percentil {p})."
        )

    return percentil_valor


# ===============================
#     PROCESAMIENTO DE IMAGEN
# ===============================


def cargar_png(path):
    """
    Carga un archivo PNG en escala de grises
    y lo devuelve como array de NumPy.
    """
    return np.array(Image.open(path).convert("L"))


def preparar_cortes_pred_gt(img_path, pred_path, gt_path):
    """
    Carga y prepara la imagen, la máscara de predicción y la máscara GT
    correspondientes a un mismo corte, aplicando la corrección geométrica
    necesaria para la predicción.
    """
    img = cargar_png(img_path)
    pred = (cargar_png(pred_path) > 0).astype(float)
    gt = (cargar_png(gt_path) > 0).astype(float)

    pred = np.rot90(pred, 1)  # Rotación correctiva

    return img, pred, gt


def normalizar_mascara_binaria(mask_path):
    """
    Normaliza y guarda una máscara binaria a valores 0 (fondo) y 1 (objeto).
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_bin = (mask > 0).astype(np.uint8)
    cv2.imwrite(mask_path, mask_bin)


def normalizar_a_uint8(imagen):
    """
    Normaliza una imagen (float32/64) al rango 0–255 y tipo uint8.
    """
    if imagen.dtype != np.uint8:
        imagen = imagen.astype(np.float32)
        imagen -= np.min(imagen)
        if np.ptp(imagen) > 0:
            imagen = 255 * (imagen / np.ptp(imagen))
        imagen = imagen.astype(np.uint8)
    return imagen


def convertir_a_bgr(imagen):
    """
    Convierte una imagen 2D o RGB a formato BGR.
    """
    imagen_uint8 = normalizar_a_uint8(imagen)
    if len(imagen_uint8.shape) == 2:  # Imagen gris
        img_bgr = cv2.cvtColor(imagen_uint8, cv2.COLOR_GRAY2BGR)
    else:  # Imagen RGB
        img_bgr = cv2.cvtColor(imagen_uint8, cv2.COLOR_RGB2BGR)
    return img_bgr


def verificar_grises(imagen):
    """
    Devuelve la imagen en escala de grises, convirtiendo si es necesario.
    """
    if imagen.ndim == 3 and imagen.shape[2] == 3:  # Imagen en color (3 canales)
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return imagen  # Ya está en grises


# ===============================
#     EVALUACIÓN DE RESULTADOS
# ===============================


def evaluar_resultados(resultados):
    """
    Evalúa el estado global de una lista de resultados parciales.
    """
    if not resultados:
        return None  # Evita fallo si la lista está vacía

    if all(r is None for r in resultados):
        return None
    elif all(r is True for r in resultados):
        return True
    else:
        return "parcial"


# =============================
#           MÉTRICAS
# =============================


def DSC(y_true, y_pred):
    """Calcula el DSC."""
    intersection = np.sum(y_true * y_pred)
    dsc = (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)

    return float(np.round(dsc, 3))


def precision(y_true, y_pred):
    """Calcula la precisión."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    prec = tp / (tp + fp + 1e-8)

    return float(np.round(prec, 3))


def recall(y_true, y_pred):
    """Calcula el recall."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    rec = tp / (tp + fn + 1e-8)

    return float(np.round(rec, 3))


def AUC(y_true, y_pred):
    """Calcula el AUC."""
    try:
        # Aplanar los arrays
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        if len(np.unique(y_true)) < 2:
            logger.warning("⚠️ AUC no definido: y_true contiene una sola clase.")
            return np.nan
        auc = float(np.round(roc_auc_score(y_true, y_pred), 3))
        return auc

    except Exception as e:
        logger.warning(f"⚠️ No se pudo calcular AUC: {e}")
        return np.nan


# ===============================
#   LOGGING DE ESTADO DE FOLDS
# ===============================


def log_estado_fold(logger, resultado, fold):
    """
    Registra en el logger el estado de ejecución de un fold
    para una etapa específica del pipeline.
    """
    if resultado is None:
        logger.skip(f"⏩ Fold {fold} ya existente.")
    elif resultado is True or isinstance(resultado, (dict, list)):
        logger.info(f"🆗 Fold {fold} completado.")
    elif resultado == "parcial":
        logger.info(f"🔁 Fold {fold} parcialmente actualizado.")
    else:
        logger.warning(f"⚠️ Fold {fold}: estado desconocido.")
