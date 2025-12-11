"""
Script: visualizar_prediccion_corte.py

Descripción:
    Genera una figura para visualizar la predicción hecha por el modelo sobre un
    corte específico de un paciente y compararla con la máscara ground truth,
    superponiendo ambas sobre la imagen original. Si no se indica un corte concreto,
    el script evalúa todos los cortes disponibles, calcula el DSC de cada uno y
    visualiza únicamente el corte con mejor desempeño.

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

    --corte (int, opcional)
        Número exacto del corte a visualizar.
        Si no se especifica, el script recorre todos los cortes del paciente
        y selecciona automáticamente aquel con máximo DSC.

    --limpiar (flag, opcional)
        Si existe una figura previa, la elimina antes de generar una nueva.

Uso por CLI:
    python -m yolo_mslesseg.extras.visualizar_prediccion_corte \
        --paciente_id P14 \
        --plano sagital \
        --modalidad FLAIR \
        --num_cortes P50 \
        --mejora HE \
        --epochs 50 \
        --k_folds 5

Entradas:
    - Imagen del corte seleccionado en formato PNG.
    - Máscaras predichas y ground truth en formato PNG.

Salidas:
    - Imagen con la visualización generada en formato PNG.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from yolo_mslesseg.utils.Modelo import Modelo
from yolo_mslesseg.utils.Paciente import Paciente
from yolo_mslesseg.utils.configurar_logging import get_logger
from yolo_mslesseg.utils.utils import (
    calcular_fold,
    int_o_percentil,
    DSC,
    crear_directorio,
    construir_nombre_configuracion,
    ruta_existente,
    paths_paciente,
    obtener_cortes_paciente,
    preparar_cortes_pred_gt,
)

logger = get_logger(__file__)

# =============================
#       FUNCIONES BASE
# =============================


def normalizar_img(img_array):
    """
    Normaliza la imagen al rango [0, 1],
    evitando divisiones por cero.
    """
    img_array = img_array.astype(float)
    return (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)


def validar_shapes(pred_array, gt_array):
    """
    Valida que las máscaras de predicción y ground truth
    tengan la misma forma.
    """
    if pred_array.shape != gt_array.shape:
        raise RuntimeError(
            f"Formas incompatibles: pred {pred_array.shape}, GT {gt_array.shape}"
        )


# =============================
#     EXTRACCIÓN DE CORTES
# =============================


def cargar_y_procesar_corte(paciente, modelo, corte):
    """
    Carga y procesa los datos correspondientes a un corte:
        - imagen normalizada,
        - máscara de predicción rotada,
        - máscara ground truth,
        - DSC entre predicción y ground truth.

    Devuelve todo en forma de tupla
    (img_array, pred_array, gt_array, dsc).
    """
    paths = paths_paciente(paciente=paciente, modelo=modelo, corte=corte)

    # Validar que existan los archivos
    for tipo, ruta in paths.items():
        if not ruta_existente(ruta):
            raise RuntimeError(
                f"Falta el archivo '{tipo}' para el corte {corte} del paciente {paciente.id}:\n{ruta}"
            )

    img_array, pred_array, gt_array = preparar_cortes_pred_gt(
        img_path=paths["img"], pred_path=paths["pred"], gt_path=paths["gt"]
    )

    img_array = normalizar_img(img_array)
    validar_shapes(pred_array, gt_array)

    dsc = DSC(pred_array, gt_array)

    return img_array, pred_array, gt_array, dsc


def seleccionar_mejor_corte(paciente, modelo):
    """
    Evalúa todos los cortes disponibles del paciente y selecciona
    aquel con mayor DSC entre la predicción y el ground truth.

    Devuelve:
        (mejor_corte, mejor_dsc, img_array, gt_array, pred_array)
    correspondientes al corte con mejor desempeño.
    """
    cortes = obtener_cortes_paciente(paciente, modelo)

    if not cortes:
        raise RuntimeError(
            f"No se encontraron cortes PNG para el paciente {paciente.id}."
        )

    mejor_corte = None
    mejor_dsc = -1.0
    mejor_img = mejor_pred = mejor_gt = None

    for corte in cortes:
        img_array, pred_array, gt_array, dsc = cargar_y_procesar_corte(
            paciente=paciente, modelo=modelo, corte=corte
        )

        if dsc > mejor_dsc:
            mejor_dsc = dsc
            mejor_corte = corte
            mejor_img = img_array
            mejor_pred = pred_array
            mejor_gt = gt_array

    return mejor_corte, mejor_dsc, mejor_img, mejor_gt, mejor_pred


# =============================
#     GENERACIÓN DE FIGURA
# =============================


def generar_figura(img_array, gt_array, pred_array, salida, corte, titulo=None):
    """
    Genera una figura superponiendo:
        - Ground Truth (verde)
        - Predicción (naranja)
        - Intersección (azul)
    sobre la imagen original.
    """
    # Máscaras TP / FP / FN
    tp = (pred_array == 1) & (gt_array == 1)  # Intersección (TP)
    fp = (pred_array == 1) & (gt_array == 0)  # Solo predicción (FP)
    fn = (pred_array == 0) & (gt_array == 1)  # Solo GT (FN)

    # Figura base
    h, w = img_array.shape
    fig_w = 6.0
    fig_h = fig_w * (h / w)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.tight_layout(pad=0)

    # Imagen base
    ax.imshow(img_array, cmap="gray", vmin=0, vmax=1)

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

    # Titulo
    if titulo:
        ax.text(
            0.5,
            0.98,
            titulo,
            transform=ax.transAxes,
            ha="center",
            va="top",
            color="white",
            fontsize=22,
            fontweight="bold",
            fontname="Arial",
        )

    # Número de corte
    ax.text(
        0.01,
        0.02,
        f"Corte {corte}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="white",
        fontsize=18,
        fontweight="bold",
        fontname="Arial",
    )

    # Leyenda
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

    plt.savefig(salida, dpi=300, pad_inches=0)
    plt.close(fig)


# =============================
#        PROCESAMIENTO
# =============================


def visualizar_mejor_corte(paciente, modelo, output_dir, limpiar):
    """
    Selecciona automáticamente el corte con mejor DSC para el paciente y
    genera la figura correspondiente en el directorio de salida.
    Si se pasa `limpiar, elimina la figura previa antes de generar una nueva.
    """
    str_mejora = paciente.mejora if paciente.mejora is not None else "Control"
    logger.info(
        f"🔎 Buscando mejor corte para paciente {paciente.id} ({str_mejora}, {paciente.plano})."
    )

    mejor_corte, mejor_dsc, img_array, gt_array, pred_array = seleccionar_mejor_corte(
        paciente=paciente, modelo=modelo
    )

    logger.info(f"🏅 Mejor corte encontrado: {mejor_corte}  (DSC = {mejor_dsc:.3f}).")

    output_path = (
        output_dir / f"{paciente.id}_{paciente.modalidad_str}_{mejor_corte}.png"
    )

    if limpiar and ruta_existente(output_path):
        logger.info(f"♻️ Limpiando figura previa.")
        output_path.unlink(missing_ok=True)

    titulo = (
        f"{paciente.id} – {str_mejora} – {paciente.plano.capitalize()}\n"
        f"(DSC = {mejor_dsc:.3f})"
    )

    generar_figura(
        img_array=img_array,
        gt_array=gt_array,
        pred_array=pred_array,
        titulo=titulo,
        corte=mejor_corte,
        salida=output_path,
    )

    logger.info(f"✅ Figura generada correctamente.")


def visualizar_corte_especifico(paciente, modelo, corte, output_dir, limpiar):
    """
    Genera la figura asociada a un corte específico del paciente.
    Calcula el DSC correspondiente y guarda la visualización en el
    directorio de salida. Si se pasa `limpiar, elimina la figura
    previa antes de generar una nueva.
    """
    str_mejora = paciente.mejora if paciente.mejora is not None else "Control"

    img_array, pred_array, gt_array, dsc = cargar_y_procesar_corte(
        paciente=paciente, modelo=modelo, corte=corte
    )

    titulo = (
        f"{paciente.id} – {str_mejora} – {paciente.plano.capitalize()}\n"
        f"(DSC = {dsc:.3f})"
    )

    output_path = output_dir / f"{paciente.id}_{paciente.modalidad_str}_{corte}.png"

    if limpiar and ruta_existente(output_path):
        logger.warning(f"♻️ Limpiando figura previa.")
        output_path.unlink(missing_ok=True)

    generar_figura(
        img_array=img_array,
        gt_array=gt_array,
        pred_array=pred_array,
        titulo=titulo,
        corte=corte,
        salida=output_path,
    )

    logger.info(f"✅ Figura generada correctamente.")


# =============================
#       FLUJO PRINCIPAL
# =============================


def ejecutar_flujo(paciente, modelo, epochs, corte, limpiar):
    """
    Ejecuta el flujo de visualización del corte especificado o,
    si no se indica corte, selecciona automáticamente el mejor
    corte para el paciente.
    """
    logger.header(f"\n🖼️ Generando visualización de predicción")

    root = Path.cwd()  # respeta demo/ si viene desde ejecutar_demo.py
    configuracion_global = construir_nombre_configuracion(modelo, epochs)

    paciente_id = paciente.id
    plano = paciente.plano
    mejora = paciente.mejora if paciente.mejora else "Control"
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

    # Delegar según el modo
    if corte is None:
        visualizar_mejor_corte(
            paciente=paciente, modelo=modelo, output_dir=output_dir, limpiar=limpiar
        )
    else:
        visualizar_corte_especifico(
            paciente=paciente,
            modelo=modelo,
            corte=corte,
            output_dir=output_dir,
            limpiar=limpiar,
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
        description="Generar figura para visualizar la predicción y el ground truth sobre un corte de un paciente."
    )
    parser.add_argument(
        "--paciente_id",
        type=str,
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
        "--corte",
        type=int,
        metavar="<corte>",
        help="Corte específico sobre el que generar la visualización.",
    )
    parser.add_argument(
        "--limpiar",
        action="store_true",
        default=False,
        help="Limpiar la figura generada previamente.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    """
    Entrada CLI del script: parsea argumentos, construye
    Modelo y Paciente y ejecuta la visualización.
    """

    args = parsear_args(argv)

    modelo = Modelo(
        plano=args.plano,
        num_cortes=args.num_cortes,
        modalidad=args.modalidad,
        k_folds=args.k_folds,
        mejora=args.mejora,
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
        corte=args.corte,
        limpiar=args.limpiar,
    )


if __name__ == "__main__":
    main()
