"""
Script: generar_gif_predicciones.py

Descripción:
    Genera un GIF animado que recorre todos los cortes disponibles de un paciente
    y superpone la máscara de predicción (rojo) y la máscara ground truth (verde).

Argumentos CLI:
    --paciente_id (str, requerido)
        ID del paciente a visualizar.

    --plano (str, requerido)
        Plano anatómico de extracción ('axial', 'coronal', 'sagital').

    --modalidad (list[str], opcional)
        Modalidad o modalidades de imagen MRI ('T1', 'T2', 'FLAIR').
        Por defecto todas.

    --num_cortes (int_o_percentil, requerido)
        Número de cortes extraídos (valor entero o percentil, por ejemplo 50 o 'P50').

    --mejora (str, opcional)
        Algoritmo de mejora aplicado ('HE', 'CLAHE', 'GC', 'LT', o None).
        Por defecto None.

    --epochs (int, requerido)
        Número de épocas del modelo entrenado.

    --k_folds (int, opcional)
        Número de folds para validación cruzada.
        Por defecto 5.

    --limpiar (flag, opcional)
        Si existe un GIF previo, lo elimina antes de generar uno nuevo.

Uso por CLI:
    python -m yolo_mslesseg.extras.generar_gif_predicciones \
        --paciente_id P14 \
        --plano sagital \
        --modalidad FLAIR \
        --num_cortes P50 \
        --mejora HE \
        --epochs 50 \
        --k_folds 5

Entrada:
    - Imagenes, máscaras predichas y ground truth en formato PNG.

Salida:
    - GIF animado con todas las predicciones del paciente.
"""

import argparse
import logging
import sys
from io import BytesIO
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from yolo_mslesseg.utils.Modelo import Modelo
from yolo_mslesseg.utils.Paciente import Paciente
from yolo_mslesseg.utils.configurar_logging import get_logger
from yolo_mslesseg.utils.utils import (
    int_o_percentil,
    calcular_fold,
    ruta_existente,
    construir_nombre_configuracion,
    crear_directorio,
    paths_paciente,
    obtener_cortes_paciente,
    preparar_cortes_pred_gt,
)

logger = get_logger(__file__)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# =============================
#       FUNCIONES BASE
# =============================


def normalizar_img_global(img_array, vmin, vmax):
    """
    Normaliza una imagen al rango [0, 1] usando un valor mínimo y máximo
    globales (compartidos entre cortes), evitando divisiones por cero.
    """
    img_array = img_array.astype(float)
    denom = vmax - vmin
    if denom <= 0:
        denom = 1.0
    return (img_array - vmin) / (denom + 1e-8)


def cargar_series_cortes(paciente, modelo):
    """
    Carga y valida todos los cortes disponibles del paciente para el modelo dado.
    Devuelve tres listas paralelas:
        - imagenes: lista de tuplas (corte, img_array)
        - preds: lista de máscaras de predicción
        - gts: lista de máscaras ground truth
    """
    cortes = obtener_cortes_paciente(paciente, modelo)

    if not cortes:
        raise RuntimeError(
            f"No se encontraron cortes PNG para el paciente {paciente.id}."
        )

    imagenes, preds, gts = [], [], []

    for corte in cortes:
        paths = paths_paciente(paciente, modelo, corte)

        # Validar existencia de archivos
        for tipo, ruta in paths.items():
            if not ruta_existente(ruta):
                raise RuntimeError(
                    f"Falta el archivo '{tipo}' para el corte {corte} del paciente "
                    f"{paciente.id}: {ruta}"
                )

        img_array, pred_array, gt_array = preparar_cortes_pred_gt(
            img_path=paths["img"],
            pred_path=paths["pred"],
            gt_path=paths["gt"],
        )

        imagenes.append((corte, img_array))
        preds.append(pred_array)
        gts.append(gt_array)

    return imagenes, preds, gts


def calcular_rango_global(imagenes):
    """
    Calcula el mínimo y máximo global de intensidad a partir
    de la lista de imágenes [(corte, img_array), ...].
    """
    global_min = min(img.min() for _, img in imagenes)
    global_max = max(img.max() for _, img in imagenes)
    return global_min, global_max


# =============================
#     GENERACIÓN DE FIGURA
# =============================


def crear_frame(img_array, pred_array, gt_array, corte, paciente, mejora, vmin, vmax):
    """
    Genera un frame para el GIF superponiendo:
        - Ground Truth (verde)
        - Predicción (naranja)
        - Intersección (azul)
    sobre la imagen original.
    """
    str_mejora = mejora if mejora is not None else "Base"

    # Normalización global
    norm = (img_array - vmin) / (vmax - vmin + 1e-8)

    # Figura cuadrada sin bordes
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    ax.axis("off")
    fig.patch.set_facecolor("black")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.margins(0, 0)
    ax.set_position([0, 0, 1, 1])

    # Imagen base
    ax.imshow(norm, cmap="gray", vmin=0, vmax=1)

    # Máscaras TP / FP / FN
    tp = (pred_array == 1) & (gt_array == 1)  # Intersección (TP)
    fp = (pred_array == 1) & (gt_array == 0)  # Solo predicción (FP)
    fn = (pred_array == 0) & (gt_array == 1)  # Solo GT (FN)

    # Superposición de máscaras
    # Orden correcto: FN → FP → TP
    ax.imshow(
        np.ma.masked_where(fn == 0, fn), cmap=ListedColormap(["#0099FF"]), alpha=0.5
    )  # Verde
    ax.imshow(
        np.ma.masked_where(fp == 0, fp), cmap=ListedColormap(["#FF4500"]), alpha=0.5
    )  # Naranja
    ax.imshow(
        np.ma.masked_where(tp == 0, tp), cmap=ListedColormap(["#00CC66"]), alpha=0.7
    )  # Azul

    # Título (arriba, centrado)
    ax.text(
        0.5,
        0.985,
        f"{paciente.id} – {str_mejora} – {paciente.plano.capitalize()}",
        ha="center",
        va="top",
        color="white",
        fontsize=22,
        fontweight="bold",
        family="Arial",
        transform=ax.transAxes,
    )

    # Número de corte (abajo, izquierda)
    ax.text(
        0.01,
        0.005,  # casi tocando el borde
        f"Corte {corte}",
        ha="left",
        va="bottom",
        color="white",
        fontsize=16,
        fontweight="bold",
        family="Arial",
        transform=ax.transAxes,
    )

    # Leyenda (abajo, derecha)
    ax.legend(
        handles=[
            mpatches.Patch(color="#00CC66", label="Intersección (TP)"),
            mpatches.Patch(color="#FF4500", label="Predicción (FP)"),
            mpatches.Patch(color="#0099FF", label="Ground Truth (FN)"),
        ],
        loc="lower right",
        prop={"family": "Arial", "weight": "bold", "size": 10},
        frameon=True,
        facecolor="black",
        edgecolor="white",
        labelcolor="white",
        framealpha=0.6,
    )

    # Guardar frame
    buf = BytesIO()
    fig.savefig(
        buf, format="png", dpi=120, pad_inches=0, facecolor="black", bbox_inches="tight"
    )
    plt.close(fig)

    buf.seek(0)
    return Image.open(buf)


def construir_frames_gif(paciente, imagenes, preds, gts, vmin, vmax):
    """
    Construye la lista de frames para el GIF a partir de las
    imágenes normalizadas globalmente y sus máscaras asociadas.
    """
    frames = []

    for (corte, img), pred, gt in zip(imagenes, preds, gts):
        frame = crear_frame(
            img_array=img,
            pred_array=pred,
            gt_array=gt,
            corte=corte,
            paciente=paciente,
            mejora=paciente.mejora,
            vmin=vmin,
            vmax=vmax,
        )
        frames.append(frame)

    return frames


# =============================
#        PROCESAMIENTO
# =============================


def generar_gif(paciente, modelo, output_path):
    """
    Genera un GIF recorriendo todos los cortes del paciente, superponiendo
    la predicción y el ground truth, con normalización global de intensidades
    y FPS ajustado al número de cortes.
    """
    # Cargar y validar todos los cortes
    imagenes, preds, gts = cargar_series_cortes(paciente, modelo)

    # Normalización global
    global_min, global_max = calcular_rango_global(imagenes)

    # Construir frames
    frames = construir_frames_gif(
        paciente=paciente,
        imagenes=imagenes,
        preds=preds,
        gts=gts,
        vmin=global_min,
        vmax=global_max,
    )

    if not frames:
        raise RuntimeError(
            f"No se pudieron generar frames para el paciente {paciente.id}."
        )

    # Establecer duración
    fps = max(3, min(12, len(frames) // 4))
    duration_ms = int(1000 / fps)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


# =============================
#       FLUJO PRINCIPAL
# =============================


def ejecutar_flujo(paciente, modelo, epochs, limpiar):
    """
    Genera el GIF completo combinando todos los cortes del paciente.
    """
    logger.header("\n🎥️ Generando GIF de predicciones")

    root = Path.cwd()  # respeta demo/ si viene desde ejecutar_demo.py
    configuracion_global = construir_nombre_configuracion(modelo, epochs)

    paciente_id = paciente.id
    plano = paciente.plano
    mejora = paciente.mejora if paciente.mejora else "Base"
    k_folds = modelo.k_folds
    fold_paciente = calcular_fold(paciente_id, k_folds)

    output_dir = (
        root
        / "visualizaciones"
        / mejora
        / configuracion_global
        / f"fold{fold_paciente}"
        / paciente_id
        / plano
    )
    crear_directorio(output_dir)

    output_path = output_dir / f"{paciente.id}_{modelo.modalidad_str}.gif"

    # Limpiar si corresponde
    if limpiar and ruta_existente(output_path):
        logger.info(f"♻️ Eliminando GIF previo.")
        output_path.unlink()

    generar_gif(paciente=paciente, modelo=modelo, output_path=output_path)

    logger.info("✅ GIF generado correctamente.")


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
        description=(
            "Generar un GIF con todos los cortes de un paciente, "
            "superponiendo la predicción y la ground truth "
            "sobre la imagen original."
        )
    )
    parser.add_argument(
        "--paciente_id",
        type=str,
        required=True,
        metavar="<paciente_id>",
        help="ID del paciente a visualizar.",
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
        help="Eliminar el GIF previo antes de generar uno nuevo.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    """
    Entrada CLI del script: parsea argumentos, construye
    Modelo y Paciente y ejecuta la generación del GIF.
    """
    args = parsear_args(argv)

    modelo = Modelo(
        plano=args.plano,
        num_cortes=args.num_cortes,
        modalidad=args.modalidad,
        mejora=args.mejora,
        k_folds=args.k_folds,
    )

    paciente = Paciente(
        id=args.paciente_id,
        plano=modelo.plano,
        modalidad=args.modalidad,
        mejora=modelo.mejora,
    )

    ejecutar_flujo(
        paciente=paciente,
        modelo=modelo,
        epochs=args.epochs,
        limpiar=args.limpiar,
    )


if __name__ == "__main__":
    main()
