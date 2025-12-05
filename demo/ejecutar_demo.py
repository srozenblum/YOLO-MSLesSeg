"""
Script: ejecutar_demo.py

Descripción:
    Ejecuta una demostración resumida y controlada del pipeline
    YOLO-MSLesSeg utilizando únicamente pacientes específicos y
    sin entrenar ningún modelo.

    Actualmente incluye dos ejecuciones individuales:
    # TODO: ACLARAR QUE SE CALCULARON CON EL SCRIPT
        - P14 sin mejora (plano sagital)
        - P18 con HE (plano sagital)

Modo de ejecución:
    Este script debe ejecutarse únicamente por CLI. No es parte del pipeline, por lo que
    no está preparado para uso interno.

Argumentos CLI:

Uso por CLI:


"""

import os
import sys
from pathlib import Path

from yolo_mslesseg.ejecutar_pipeline import main as pipeline_main


def ejecutar_demo_paciente(paciente_id, mejora, plano):
    """
    Ejecuta la demo para un paciente específico utilizando el pipeline real.

    Parámetros:
        paciente_id (str): Identificador del paciente (por ejemplo, "P14").
        mejora      (str opcional): Algoritmo de mejora de imagen ("HE", "CLAHE",
                                    "GC", "LT") o None para sin mejora.
        plano       (str): Plano anatómico ("axial", "coronal", "sagital").

    Descripción:
        - Imprime un encabezado en stderr para mantener consistencia con el
          sistema de logging del pipeline.
        - Construye un `argv` equivalente a una llamada CLI al pipeline.
        - Invoca `pipeline_main(argv)` como si fuera una ejecución independiente.
    """
    sys.stderr.write(
        f"\n=== Ejecutando demo para {paciente_id} "
        f"(mejora={mejora}, plano={plano}) ===\n"
    )

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


def main():
    # Guardar cwd original
    original_cwd = Path.cwd()
    demo_cwd = Path(__file__).resolve().parent

    # Cambiar cwd al demo
    os.chdir(demo_cwd)

    try:
        ejecutar_demo_paciente("P14", mejora=None, plano="sagital")
        ejecutar_demo_paciente("P18", mejora="HE", plano="sagital")

    finally:
        # Restaurar cwd original
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
