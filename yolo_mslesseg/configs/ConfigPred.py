from pathlib import Path

from yolo_mslesseg.utils.configurar_logging import get_logger
from yolo_mslesseg.utils.utils import (
    ruta_existente,
    crear_directorio,
    eliminar_directorio,
    listar_pacientes,
    existe_modelo_entrenado,
    calcular_fold,
)

# Configurar logger
logger = get_logger(__file__)


class ConfigPred:
    """
    Clase: ConfigPred

    Descripción:
        Configuración y gestión de rutas/variables globales para la generación de
        predicciones, implementada en `generar_predicciones.py`. Se encarga de
        verificar, crear y limpiar los directorios necesarios tanto para un fold
        como para un paciente individual, asegurando la correcta lectura de imágenes
        para la generación de las máscaras predichas.

    Casos de uso contemplados:
        1. Modo paciente individual (`paciente` ≠ None):
           Genera predicciones solo para el paciente especificado.

        2. Modo fold (`paciente` = None, `fold_test` ≠ None):
           Genera predicciones para todos los pacientes del fold indicado.

    Convenciones de directorios:
        datasets/: conjunto de datos dividido por folds y pacientes
        └── <mejora>/
             └── <modalidad>_<num_cortes>c_<k_folds>folds_<epochs>epochs/
                 ├── <fold_test>/
                 │   ├── PX/
                 │   │   └── <plano>
                 │   │        ├── images/: imágenes de entrada
                 │   │        ├── labels/: anotaciones YOLO
                 │   │        ├── GT_masks/: máscaras ground truth asociadas a cada imagen
                 │   │        └── pred_masks/: máscaras 2D predichas por el modelo
                 │   └── ...
                 └── ...

        trains/: resultados del entrenamiento (pesos YOLO)

    Atributos:
        modelo (Modelo):
            Instancia del modelo, que define plano, modalidades, mejora y base_path.

        plano (str):
            Plano anatómico de procesamiento ('axial', 'coronal', 'sagital').
            Coincide con `modelo.plano`.

        epochs (int):
            Número de épocas del modelo YOLO entrenado.

        k_folds (int):
            Número de folds para validación cruzada.

        paciente (Paciente, opcional):
            Instancia del paciente si la ejecución es individual.
            Es None en ejecución de fold.

        fold_test (int, opcional):
            Número de fold utilizado como conjunto de test (1, ..., `k_folds`).
            Si la ejecución es para un paciente individual, se calcula automáticamente.

        dataset_fold_dir (Path):
             Directorio del fold correspondiente dentro del dataset.

        train_base_dir (Path):
            Directorio base del entrenamiento YOLO para el fold correspondiente.

        # --- Atributos válidos solo en modo paciente ---
        paciente_root (Path, opcional):
            Directorio base del paciente dentro de su fold.

        paciente_dir (dict[str, Path], opcional):
            Diccionario de subdirectorios del paciente:
                - images/
                - pred_masks/
    """

    def __init__(
        self,
        modelo,
        epochs,
        k_folds=5,
        fold_test=None,
        paciente=None,
    ):
        # --- Atributos principales ---
        self._set_atributos_principales(
            modelo=modelo,
            epochs=epochs,
            k_folds=k_folds,
            paciente=paciente,
            fold_test=fold_test,
        )

        # --- Determinar modo de ejecución ---
        self._resolver_modo_ejecucion()

        # --- Directorios del dataset (entrada y salida) ---
        self._resolver_rutas_dataset()

        # --- Directorios de entrenamiento (entrada) ---
        self._resolver_rutas_entrenamiento()

        # --- Rutas específicas del paciente (si aplica) ---
        self._resolver_rutas_paciente()

    # ======================================
    #   MÉTODOS AUXILIARES DEL CONSTRUCTOR
    # ======================================

    def _set_atributos_principales(self, modelo, epochs, k_folds, paciente, fold_test):
        self.modelo = modelo
        self.plano = modelo.plano
        self.epochs = epochs
        self.k_folds = k_folds
        self.paciente = paciente
        self.fold_test = fold_test

    def _resolver_modo_ejecucion(self):
        self.es_paciente_individual = self.paciente is not None
        self.es_fold = not self.es_paciente_individual and self.fold_test is not None

        if self.es_paciente_individual:
            # Determinar fold asignado al paciente
            self.fold_test = calcular_fold(
                paciente_id=self.paciente.id,
                k_folds=self.k_folds,
            )
            return

        elif self.es_fold:
            return  # Nada más que hacer

        else:
            raise ValueError(
                "Debe especificarse un modo de ejecución: fold de test o paciente individual."
            )

    def _resolver_rutas_dataset(self):
        self.dataset_base_dir = Path("datasets") / f"{self.modelo.base_path}"

        # Dataset del fold
        self.dataset_fold_dir = self.dataset_base_dir / f"fold{self.fold_test}"

    def _resolver_rutas_entrenamiento(self):
        # Directorio base del entrenamiento
        self.train_base_dir = (
            Path("trains")
            / f"{self.modelo.base_path}_{self.epochs}epochs"
            / self.plano
            / f"fold{self.fold_test}"
        )

        # Archivo de pesos YOLO entrenado
        self.model_path = self.train_base_dir / "weights" / "best.pt"

    def _resolver_rutas_paciente(self):
        if not self.es_paciente_individual:
            return

        self.paciente_root = (
            self.dataset_base_dir
            / f"fold{self.fold_test}"
            / self.paciente.id
            / self.plano
        )

        # Diccionario con los subdirectorios
        self.paciente_dir = {
            subdir: self.paciente_root / subdir for subdir in ["images", "pred_masks"]
        }

    # =============================
    #           LIMPIEZA
    # =============================

    def _limpiar_predicciones_fold(self):
        """
        Limpia la carpeta pred_masks/ en el plano correspondiente
        para todos los pacientes del fold.
        """
        if ruta_existente(self.dataset_fold_dir):
            pacientes = listar_pacientes(self.dataset_fold_dir)

            for paciente_id in pacientes:
                pred_dir = (
                    self.dataset_fold_dir / paciente_id / self.plano / "pred_masks"
                )
                if ruta_existente(pred_dir):
                    try:
                        eliminar_directorio(pred_dir)
                    except Exception as e:
                        logger.warning(f"⚠️ No se pudo eliminar {pred_dir}: {e}")

    def _limpiar_predicciones_paciente(self):
        """
        Limpia los cortes predichos en el plano correspondiente
        para un paciente individual.
        """
        if ruta_existente(self.paciente_dir["pred_masks"]):
            try:
                eliminar_directorio(self.paciente_dir["pred_masks"])
            except Exception as e:
                logger.warning(
                    f"⚠️ No se pudo eliminar {self.paciente_dir['pred_masks']}: {e}"
                )

    def limpiar_predicciones(self):
        """
        Limpia los cortes predichos según el plano del modelo
        y el modo de ejecución.

        - Modo fold:
          Limpia las predicciones de todos los pacientes del fold.

        - Modo paciente individual:
          Limpia únicamente las predicciones del paciente especificado,
          sin afectar al resto del fold.
        """
        if self.es_fold:
            self._limpiar_predicciones_fold()

        elif self.es_paciente_individual:
            self._limpiar_predicciones_paciente()

        else:
            raise ValueError("Debe especificarse un fold o un paciente.")

    # =============================
    #         VERIFICACIÓN
    # =============================

    def _verificar_paths_fold(self):
        """
        Verifica que existan los archivos de entrada y el directorio de salida para
        los pacientes del fold.
        - Entrada: directorio de imágenes por paciente (images_dir).
        - Salida: directorio de máscaras 2D predichas por paciente (pred_masks_dir).
        """
        pacientes = listar_pacientes(self.dataset_fold_dir)

        for paciente_id in pacientes:
            paciente_dir = self.dataset_fold_dir / paciente_id / self.plano
            if paciente_dir.is_dir():
                paciente_images_dir = paciente_dir / "images"
                paciente_pred_masks_dir = paciente_dir / "pred_masks"

                # images_dir
                if not ruta_existente(
                    paciente_images_dir
                ):  # Lanza excepción si no existe
                    raise FileNotFoundError(
                        f"No existe el directorio de imágenes del paciente {paciente_id}: {paciente_images_dir}."
                    )

                # pred_masks_dir
                crear_directorio(paciente_pred_masks_dir)  # Asegurar salida

    def _verificar_paths_paciente(self):
        """
        Verifica que existan los directorios de entrada y salida para un paciente individual.
        - Entrada: directorio de imágenes (paciente_dir["images"]).
        - Salida: directorio de máscaras 2D predichas (paciente_dir["pred_masks"]).
        """
        # paciente_dir["images"]
        if not ruta_existente(
            self.paciente_dir["images"]
        ):  # Lanza excepción si no existe
            raise FileNotFoundError(
                f"No existe el directorio de imágenes del paciente {self.paciente.id}: {self.paciente_dir['images']}."
            )

        # paciente_dir["pred_masks"]
        crear_directorio(self.paciente_dir["pred_masks"])  # Asegurar salida

    def _verificar_path_modelo(self):
        """
        Verifica que exista el archivo de pesos del modelo YOLO entrenado.
        """
        if not existe_modelo_entrenado(
            modelo=self.modelo, epochs=self.epochs, fold_test=self.fold_test
        ):  # Lanza excepción si no existe
            raise FileNotFoundError(
                f"No existe el modelo entrenado en {self.model_path}."
            )

    def verificar_paths(self):
        """
        Verifica que existan los directorios de entrada y salida para
        la generación de predicciones.

        - Verifica siempre la existencia del archivo del modelo entrenado.

        - Modo fold:
            * Verifica las rutas de todos los pacientes del fold.

        - Modo paciente individual:
            * Verifica únicamente las rutas del paciente especificado.
        """
        self._verificar_path_modelo()

        if self.es_fold:
            self._verificar_paths_fold()

        else:  # es_paciente_individual
            self._verificar_paths_paciente()

    # =============================
    #        REPRESENTACIÓN
    # =============================

    def __repr__(self):
        """Representación de la instancia de ConfigPred."""
        if self.es_paciente_individual:
            return f"{self.__class__.__name__}(modelo={self.modelo.model_string}, epochs={self.epochs}, paciente={self.paciente.id})"
        else:
            return f"{self.__class__.__name__}(modelo={self.modelo.model_string}, epochs={self.epochs}, fold={self.fold_test})"
