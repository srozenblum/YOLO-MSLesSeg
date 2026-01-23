"""
Script: paths.py

Descripción:
    Define y centraliza las rutas absolutas del proyecto YOLO-MSLesSeg.
    Este módulo actúa como fuente única para la localización del dataset
    MSLesSeg, garantizando independencia del directorio de ejecución
    (cwd) y consistencia entre pipeline completo y demo.

Convenciones de uso:
    - Todas las clases y scripts que necesiten acceder al dataset MSLesSeg
      deben importar DATASET_MSLESSEG desde este módulo.
"""

from pathlib import Path

# Ruta a la raíz del repositorio
REPO_ROOT = Path(__file__).resolve().parents[2]

# Ruta al dataset de entrada (conjunto de entrenamiento)
DATASET_MSLESSEG = REPO_ROOT / "MSLesSeg-Dataset" / "train"

if not DATASET_MSLESSEG.is_dir():  # Validación estricta
    raise RuntimeError(f"Dataset no encontrado: {DATASET_MSLESSEG}")
