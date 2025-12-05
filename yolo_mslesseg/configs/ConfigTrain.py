from pathlib import Path

from yolo_mslesseg.utils.configurar_logging import get_logger
from yolo_mslesseg.utils.utils import (
    ruta_existente,
    crear_directorio,
    eliminar_directorio,
)

# Configurar logger
logger = get_logger(__file__)


class ConfigTrain:
    """
    Clase: ConfigTrain

    Descripción:
        Configuración y gestión de rutas/variables globales para el entrenamiento
        del modelo YOLO, implementado en `train.py`. Se encarga de verificar, crear
        y limpiar los directorios asociados a cada fold y plano, asegurando la
        disponibilidad del dataset y del archivo YAML para un correcto entrenamiento
        del modelo.

    Convenciones de directorios:
        datasets/: conjunto de datos dividido por folds y pacientes
        └── <mejora>/
             └── <modalidad>_<num_cortes>c_<k_folds>folds_<epochs>epochs/
                 ├── <fold_test>/
                 │   ├── PX/
                 │   │   └── <plano>
                 │   │        ├── images/: imágenes de entrada
                 │   │        ├── labels/: anotaciones YOLO
                 │   │        └── GT_masks/: máscaras ground truth asociadas a cada imagen
                 │   └── ...
                 └── ...

        trains/: resultados del entrenamiento YOLO
        └── <modalidad>_<num_cortes>c_<k_folds>folds_<epochs>epochs/
            └── <plano>/
                ├── <fold_test>/
                │    ├── weights/
                │    └── salidas adicionales del entrenamiento
                └── ...

        yaml_files/: archivos de configuración YOLO generados para cada fold

    Atributos:
        modelo (Modelo):
            Instancia del modelo, que define plano, modalidades, mejora y base_path.

        plano (str):
            Plano anatómico de procesamiento ('axial', 'coronal' o 'sagital').
            Coincide con `modelo.plano`.

        epochs (int):
            Número de épocas de entrenamiento.

        fold_test (int):
            Número de fold utilizado como conjunto de test (1, ..., `k_folds`).

        dataset_base_dir (Path):
            Directorio base del dataset dividido en folds:
            datasets/<base_path>/.

        fold_dir (Path):
            Directorio del dataset correspondiente al fold utilizado:
            datasets/<base_path>/fold<fold_test>/.

        fold_train_dir (Path):
            Directorio temporal con las imágenes y anotaciones usadas para
            entrenamiento: datasets/<base_path>/train_fold<fold_test>/<plano>/.

        train_output_dir (Path):
            Directorio donde se almacenan los resultados del entrenamiento:
            trains/<base_path>_<epochs>epochs/<plano>/.

        yaml_path (Path):
            Archivo YAML con la configuración del dataset YOLO generado para este fold.

        weights_path (Path):
            Ruta al archivo de pesos base del modelo YOLO.
    """

    def __init__(self, modelo, epochs, fold_test):
        # --- Atributos principales ---
        self._set_atributos_principales(modelo, epochs, fold_test)

        # --- Directorios del dataset (entrada) ---
        self._resolver_rutas_dataset()

        # --- Directorios del entrenamiento (salida) ---
        self._resolver_rutas_salida()

    # ======================================
    #  MÉTODOS AUXILIARES DEL CONSTRUCTOR
    # ======================================

    def _set_atributos_principales(self, modelo, epochs, fold_test):
        self.modelo = modelo
        self.plano = modelo.plano
        self.epochs = epochs
        self.fold_test = fold_test

        if self.fold_test is None:
            raise ValueError("Debe especificarse fold_test para entrenar el modelo.")

    def _resolver_rutas_dataset(self):
        # Directorio base
        self.dataset_base_dir = Path("datasets") / f"{self.modelo.base_path}"

        # Directorio del fold en el dataset
        self.fold_dir = self.dataset_base_dir / f"fold{self.fold_test}"

        # Directorios específicos para subconjuntos train/test de cada fold
        self.fold_train_dir = (
            self.dataset_base_dir / f"train_fold{self.fold_test}" / self.plano
        )

        self.fold_test_dir = (
            self.dataset_base_dir / f"test_fold{self.fold_test}" / self.plano
        )

    def _resolver_rutas_salida(self):

        # Directorio de salida del entrenamiento (único por plano)
        self.train_output_dir = (
            Path("trains") / f"{self.modelo.base_path}_{self.epochs}epochs" / self.plano
        )

        # Archivo YAML del dataset
        self.yaml_path = (
            Path("datasets")
            / "yaml_files"
            / f"dataset_{self.modelo.model_string}_fold{self.fold_test}.yaml"
        )

        # Pesos base YOLO
        self.weights_path = Path("yolo11n-seg.pt")

    # =============================
    #           LIMPIEZA
    # =============================

    def limpiar_entrenamiento(self):
        """
        Limpia los resultados de entrenamiento.
        """
        if ruta_existente(self.train_output_dir):
            eliminar_directorio(self.train_output_dir)

    # =============================
    #         VERIFICACIÓN
    # =============================

    def verificar_paths(self):
        """
        Verifica que existan los directorios de entrada y salida para
        la generación del modelo YOLO.

        - Verifica la existencia del directorio del dataset (dataset_base_dir)
          con las imágenes, anotaciones y ground truth.

        - Verifica la existencia del directorio de salida de resultados
          (train_output_dir) y del directorio de YAML (yaml_dir), creándolos
          si no existen.
        """
        # dataset_base_dir
        if not ruta_existente(self.dataset_base_dir):  # Lanza excepción si no existe
            raise FileNotFoundError(
                f"No se encontró el dataset base: {self.dataset_base_dir}"
            )

        # yaml_dir
        yaml_dir = self.yaml_path.parent
        crear_directorio(yaml_dir)

        # train_output_dir
        crear_directorio(self.train_output_dir)  # Asegurar salida

    # =============================
    #        REPRESENTACIÓN
    # =============================

    def __repr__(self):
        """Representación de la instancia de ConfigTrain."""
        return f"{self.__class__.__name__}(modelo={self.modelo.model_string}, fold={self.fold_test}, epochs={self.epochs})"
