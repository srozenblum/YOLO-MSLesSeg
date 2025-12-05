from pathlib import Path

from yolo_mslesseg.utils.configurar_logging import get_logger
from yolo_mslesseg.utils.utils import (
    ruta_existente,
    crear_directorio,
    listar_pacientes,
    calcular_fold,
)

# Configurar logger
logger = get_logger(__file__)


class ConfigRecVol:
    """
    Clase: ConfigRecVol

    Descripción:
        Configuración y gestión de rutas/variables globales para la reconstrucción
        de volúmenes, implementada en `reconstruir_volumen.py`. Se encarga de verificar,
        crear y limpiar los directorios necesarios tanto para un fold como para un paciente
        individual, asegurando la disponibilidad de las máscaras predichas y los volúmenes
        de ground truth para la correcta generación de los volúmenes finales.

    Casos de uso contemplados:
        1. Modo paciente individual (`paciente` ≠ None)
           Reconstruye volúmenes solo para el paciente especificado.

        2. Modo fold (`paciente` = None, `fold_test` ≠ None)
           Reconstruye volúmenes para todos los pacientes del fold indicado.

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

        vols/: volúmenes predichos por el modelo (reconstruidos a partir de las máscaras 2D)
        └── <mejora>/
             └── <modalidad>_<num_cortes>c_<k_folds>folds_<epochs>epochs/
                 ├── <fold_test>/
                 │   ├── PX/
                 │   │   └── PX_<plano>.nii.gz
                 │   └── ...
                 └── ...

        GT/: volúmenes ground truth

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

        vols_base_dir (Path):
            Directorio base de los volúmenes reconstruidos del experimento.

        vols_fold_dir (Path):
            Directorio de los volúmenes reconstruidos del fold.

        gt_dir (Path):
            Directorio base de los volúmenes ground truth.

        # --- Atributos válidos solo en modo paciente ---
        paciente_pred_masks (Path, opcional):
            Directorio con las máscaras 2D predichas del paciente.

        paciente_vol_root (Path, opcional):
            Directorio base donde se almacenará el volumen reconstruido del paciente.

        paciente_pred_vol (Path, opcional):
            Ruta al archivo NIfTI del volumen reconstruido del paciente
            en el plano del modelo.

        paciente_gt_vol (Path, opcional):
            Ruta al archivo NIfTI del ground truth del paciente.
    """

    def __init__(
        self,
        modelo,
        epochs,
        k_folds=5,
        paciente=None,
        fold_test=None,
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

        # --- Directorios del dataset (entrada) ---
        self._resolver_rutas_dataset()

        # --- Directorios de volúmenes (salida) ---
        self._resolver_rutas_vols()

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

        # Directorio GT
        self.gt_dir = Path("GT") / "train"

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

    def _resolver_rutas_vols(self):
        self.vols_base_dir = (
            Path("vols") / f"{self.modelo.base_path}_{self.epochs}epochs"
        )

        # Volúmenes del fold
        self.vols_fold_dir = self.vols_base_dir / f"fold{self.fold_test}"

    def _resolver_rutas_paciente(self):
        if not self.es_paciente_individual:
            return

        self.paciente_pred_masks = (
            self.dataset_base_dir
            / f"fold{self.fold_test}"
            / self.paciente.id
            / self.plano
            / "pred_masks"
        )
        self.paciente_vol_root = (
            self.vols_base_dir / f"fold{self.fold_test}" / self.paciente.id
        )

        # Archivos NIfTI (volumen predicho y ground truth)
        self.paciente_pred_vol = (
            self.paciente_vol_root / f"{self.paciente.id}_{self.plano}.nii.gz"
        )
        self.paciente_gt_vol = (
            self.gt_dir / self.paciente.id / f"{self.paciente.id}_MASK.nii.gz"
        )

    # =============================
    #           LIMPIEZA
    # =============================

    def _limpiar_volumenes_fold(self):
        """
        Limpia los volúmenes reconstruidos en el plano correspondiente
        para todos los pacientes del fold.
        """
        if ruta_existente(self.vols_fold_dir):
            pacientes = listar_pacientes(self.vols_fold_dir)

            for paciente_id in pacientes:
                paciente_vols_dir = self.vols_fold_dir / paciente_id
                if not paciente_vols_dir.is_dir():
                    continue

                # Eliminar archivo NIfTI del plano del modelo
                for archivo in paciente_vols_dir.iterdir():
                    if self.plano.lower() in archivo.name.lower() and archivo.suffixes[
                        -2:
                    ] == [".nii", ".gz"]:
                        try:
                            archivo.unlink()
                        except Exception as e:
                            logger.warning(f"⚠️ No se pudo eliminar {archivo}: {e}")

    def _limpiar_volumen_paciente(self):
        """
        Limpia el volumen reconstruido de un paciente individual.
        """
        if ruta_existente(self.paciente_vol_root):
            try:
                self.paciente_pred_vol.unlink()
            except Exception as e:
                logger.warning(f"⚠️ No se pudo eliminar el volumen: {e}")

    def limpiar_volumenes(self):
        """
        Limpia los volúmenes reconstruidos según el plano del modelo
        y el modo de ejecución.

        - Modo fold:
          Limpia los volúmenes reconstruidos de todos los pacientes del fold.

        - Modo paciente individual:
          Limpia únicamente el volumen reconstruido del paciente especificado,
          sin afectar al resto del fold.
        """
        if self.es_fold:
            self._limpiar_volumenes_fold()

        elif self.es_paciente_individual:
            self._limpiar_volumen_paciente()

    # =============================
    #         VERIFICACIÓN
    # =============================

    def _verificar_paths_fold(self):
        """
        Verifica que existan los archivos de entrada y el directorio de salida para
        los pacientes del fold.
        - Entrada: directorio de máscaras 2D predichas por paciente (pred_masks_dir).
        - Salida: directorio de volúmenes reconstruidos por paciente (vols_fold_dir).
        """
        pacientes = listar_pacientes(self.dataset_fold_dir)

        for paciente_id in pacientes:
            paciente_pred_masks_dir = (
                self.dataset_fold_dir / paciente_id / self.plano / "pred_masks"
            )
            paciente_vols_fold_dir = self.vols_fold_dir / paciente_id

            # pred_masks_dir
            if not ruta_existente(
                paciente_pred_masks_dir
            ):  # Lanza excepción si no existe
                raise FileNotFoundError(
                    f"Faltan pred_masks de {paciente_id}: {paciente_pred_masks_dir}."
                )

            # vols_fold_dir
            crear_directorio(paciente_vols_fold_dir)  # Asegurar salida

    def _verificar_paths_paciente(self):
        """
        Verifica que existan los directorios de entrada y salida para un paciente individual.
        - Entrada: directorio de máscaras 2D predichas (paciente_pred_masks).
        - Salida: directorio de volúmenes reconstruidos (paciente_vol_root).
        """
        if not ruta_existente(self.paciente_pred_masks):  # Lanza excepción si no existe
            raise FileNotFoundError(
                f"No existen pred_masks del paciente: {self.paciente_pred_masks}"
            )
        crear_directorio(self.paciente_vol_root)  # Asegurar salida

    def _verificar_paths_gt(self):
        if not ruta_existente(self.gt_dir):
            raise FileNotFoundError(f"No existe el directorio GT/: {self.gt_dir}")

    def verificar_paths(self):
        """
        Verifica que existan los directorios de entrada y salida para
        la reconstrucción de volúmenes.

        - Verifica siempre la existencia del directorio de volúmenes ground truth.

        - Modo fold:
            * Verifica las rutas de todos los pacientes del fold.

        - Modo paciente individual:
            * Verifica únicamente las rutas del paciente especificado.
        """
        self._verificar_paths_gt()

        if self.es_fold:
            self._verificar_paths_fold()

        else:  # es_paciente_individual
            self._verificar_paths_paciente()

    # =============================
    #        REPRESENTACIÓN
    # =============================

    def __repr__(self):
        """Representación de la instancia de ConfigRecVol."""
        if self.es_paciente_individual:
            return f"{self.__class__.__name__}(modelo={self.modelo.model_string}, epochs={self.epochs}, paciente={self.paciente.id})"
        else:
            return f"{self.__class__.__name__}(modelo={self.modelo.model_string}, epochs={self.epochs}, fold={self.fold_test})"
