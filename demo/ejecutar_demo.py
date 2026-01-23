"""
Script: ejecutar_demo.py

Descripción:
    Ejecuta una demostración simplificada y controlada del pipeline
    YOLO-MSLesSeg utilizando únicamente pacientes específicos y
    sin entrenar ningún modelo.

    Incluye dos ejecuciones de pacientes individuales, seleccionados a
    partir del análisis de resultados en `analizar_pacientes_dsc.py`:
        - Paciente con mayor DSC: P14, sin algoritmo de mejora, en el plano sagital.
        - Paciente con menor DSC: P18, con ecualización de histograma (HE), en el plano axial.

    El número de cortes (`num_cortes`) se fija explícitamente para cada
    ejecución, ya que el dataset incluido en la demo no es representativo
    para el cálculo de percentiles globales.

    Además de ejecutar el pipeline, genera dos visualizaciones por cada paciente:
        - Visualización para el mejor corte: imagen estática que muestra el corte
          que obtuvo el mayor DSC, con la predicción del modelo superpuesta (TP/FP/FN).
        - GIF completo: animación dinámica que recorre todos los cortes del paciente
          que contienen lesión, con la predicción del modelo superpuesta (TP/FP/FN).

Modo de ejecución:
    Este script debe ejecutarse únicamente por CLI. No es parte del pipeline,
    por lo que no está preparado para uso interno.

Argumentos CLI:
    Todos los parámetros necesarios para la ejecución del pipeline están fijados
    dentro de este script. La demo no admite argumentos por línea de comandos:
    las configuraciones están fijadas para garantizar una ejecución reproducible
    y aislada del flujo de trabajo normal.

Uso por CLI:
    python -m demo.ejecutar_demo
"""

import os
from pathlib import Path

from yolo_mslesseg.ejecutar_pipeline import main as pipeline_main
from yolo_mslesseg.extras.generar_gif_predicciones import main as generar_gif
from yolo_mslesseg.extras.visualizar_prediccion_corte import (
    main as visualizar_prediccion_corte,
)
from yolo_mslesseg.utils.configurar_logging import get_logger, configurar_logging_demo

# Configurar logger
logger = get_logger(__file__)


def ejecutar_demo_paciente(paciente_id, mejora, plano):
    """
    Ejecuta la demo para un paciente específico utilizando el pipeline.
    """

    logger.header(f"\n🧪 Ejecutando demo de YOLO-MSLesSeg")

    argv = [
        "--plano",
        plano,
        "--modalidad",
        "FLAIR",
        "--num_cortes",
        "P50",
        "--epochs",
        "50",
        "--k_folds",
        "5",
        "--paciente_id",
        paciente_id,
        "--limpiar",
    ]

    if mejora is not None:
        argv += ["--mejora", mejora]

    pipeline_main(argv)
    generar_gif(argv)
    visualizar_prediccion_corte(argv)


def main():
    """
    Entrada CLI del script.
    """
    # Guardar cwd original
    original_cwd = Path.cwd()
    demo_cwd = Path(__file__).resolve().parent

    # Cambiar cwd al demo
    os.chdir(demo_cwd)

    # Configurar el logging de la demo (demo.log)
    configurar_logging_demo()

    try:
        ejecutar_demo_paciente("P14", mejora=None, plano="sagital")
        ejecutar_demo_paciente("P18", mejora="HE", plano="axial")

    finally:
        # Restaurar cwd original
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
