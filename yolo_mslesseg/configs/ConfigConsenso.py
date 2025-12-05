from pathlib import Path

from yolo_mslesseg.utils.configurar_logging import get_logger
from yolo_mslesseg.utils.utils import ruta_existente, listar_pacientes, calcular_fold

# Configurar logger
logger = get_logger(__file__)


class ConfigConsenso:
    """
    Clase: ConfigConsenso

    Descripción:
        Configuración y gestión de rutas/variables globales para la generación
        del volumen consenso, implementada en `generar_consenso.py`. Se encarga
        de verificar, crear y limpiar los directorios necesarios tanto para un
        fold como para un paciente individual, asegurando la correcta disponibilidad
        de los volúmenes necesarios para la generación de las máscaras de consenso.

    Casos de uso contemplados:
        1. Modo paciente individual (`paciente` ≠ None)
           Genera el consenso solo para el paciente especificado.

        2. Modo fold (`paciente` = None, `fold_test` ≠ None)
           Genera los consensos para todos los pacientes del fold indicado.

    Convenciones de directorios:
        vols/: volúmenes predichos por el modelo
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
            Plano anatómico fijo ('consenso').

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

        vols_base_dir (Path):
            Directorio base de volúmenes reconstruidos del experimento.

        vols_fold_dir (Path):
            Directorio de volúmenes reconstruidos del fold.

        gt_dir (Path):
            Directorio base de los volúmenes ground truth.

         # --- Atributos válidos solo en modo paciente ---
        paciente_vol_root (Path, opcional):
            Directorio con los volúmenes reconstruidos del paciente.

        paciente_pred_vol (dict[str, Path], opcional):
            Diccionario de rutas a los volúmenes predichos del paciente por plano.

        paciente_gt_vol (Path, opcional):
            Ruta al volumen ground truth del paciente.
    """

    PLANOS = ("axial", "coronal", "sagital", "consenso")

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

        # --- Directorios de volúmenes (entrada y salida) ---
        self._resolver_rutas_vols()

        # --- Rutas del paciente (si corresponde) ---
        self._resolver_rutas_paciente()

    # ======================================
    #   MÉTODOS AUXILIARES DEL CONSTRUCTOR
    # ======================================

    def _set_atributos_principales(self, modelo, epochs, k_folds, paciente, fold_test):
        self.modelo = modelo
        self.plano = "consenso"
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

    def _resolver_rutas_vols(self):
        self.vols_base_dir = (
            Path("vols") / f"{self.modelo.base_path}_{self.epochs}epochs"
        )

        # Volúmenes del fold
        self.vols_fold_dir = self.vols_base_dir / f"fold{self.fold_test}"

    def _resolver_rutas_paciente(self):
        if not self.es_paciente_individual:
            return

        self.paciente_vol_root = self.vols_fold_dir / self.paciente.id

        # Diccionario con los volúmenes predichos por plano
        self.paciente_pred_vol = {
            plano: self.paciente_vol_root / f"{self.paciente.id}_{plano}.nii.gz"
            for plano in self.PLANOS
        }

        # Ground truth
        self.paciente_gt_vol = (
            self.gt_dir / self.paciente.id / f"{self.paciente.id}_MASK.nii.gz"
        )

    # =============================
    #           LIMPIEZA
    # =============================

    def _limpiar_consensos_fold(self):
        """
        Limpia los archivos de consenso en el plano correspondiente
        para todos los pacientes del fold.
        """
        if ruta_existente(self.vols_fold_dir):
            pacientes = listar_pacientes(self.vols_fold_dir)

            for paciente_id in pacientes:
                paciente_dir = self.vols_fold_dir / paciente_id
                if not paciente_dir.is_dir():
                    continue

                # Buscar y eliminar solo archivos de consenso
                for archivo in paciente_dir.iterdir():
                    if "consenso" in archivo.name.lower() and archivo.name.endswith(
                        ".nii.gz"
                    ):
                        try:
                            archivo.unlink()
                        except Exception as e:
                            logger.warning(f"⚠️ No se pudo eliminar {archivo}: {e}")

    def _limpiar_consenso_paciente(self):
        """
        Limpia el archivo de consenso de un paciente individual.
        """
        consenso_path = self.paciente_pred_vol["consenso"]
        if ruta_existente(consenso_path):
            try:
                consenso_path.unlink()
            except Exception as e:
                logger.warning(f"⚠️ No se pudo eliminar consenso: {e}")

    def limpiar_consenso(self):
        """
        Limpia los volúmenes de consenso según el plano del modelo
        y el modo de ejecución.

        - Modo fold:
          Limpia los consensos de todos los pacientes del fold.

        - Modo paciente individual:
          Limpia únicamente el consenso del paciente especificado,
          sin afectar al resto del fold.
        """
        if self.es_fold:
            self._limpiar_consensos_fold()

        elif self.paciente is not None:
            self._limpiar_consenso_paciente()

    # =============================
    #         VERIFICACIÓN
    # =============================

    def _verificar_paths_fold(self):
        """
        Verifica que existan los archivos de entrada y el directorio de salida
        para los pacientes del fold.
        - Entrada: volúmenes predichos en los 3 planos.
        - Salida: mismo directorio que el de entrada, se verifica implícitamente.
        """
        pacientes = listar_pacientes(self.vols_fold_dir)

        for paciente_id in pacientes:
            paciente_root = self.vols_fold_dir / paciente_id

            # Volumen por plano
            for plano in ("axial", "coronal", "sagital"):
                vol_path = paciente_root / f"{paciente_id}_{plano}.nii.gz"
                if not ruta_existente(vol_path):  # Lanza excepción si no existe
                    raise FileNotFoundError(
                        f"Falta el volumen {plano} de {paciente_id}: {vol_path}."
                    )

    def _verificar_paths_paciente(self):
        """
        Verifica que existan los archivos de entrada y el directorio de salida para un paciente individual.
        - Entrada: volúmenes predichos en los 3 planos (paciente_pred_vol por plano).
        - Salida: mismo directorio que el de entrada, se verifica implícitamente.
        """
        # paciente_pred_vol por plano
        for plano in ("axial", "coronal", "sagital"):
            vol_path = self.paciente_vol_root / f"{self.paciente.id}_{plano}.nii.gz"
            if not ruta_existente(vol_path):  # Lanza excepción si no existe
                raise FileNotFoundError(
                    f"Falta el volumen predicho {plano} del paciente {self.paciente.id}: {vol_path}."
                )

    def _verificar_paths_gt(self):
        """
        Verifica que exista el directorio de volúmenes ground truth.
        """
        if not ruta_existente(self.gt_dir):  # Lanza excepción si no existe
            raise FileNotFoundError(f"No existe el directorio de GT: {self.gt_dir}")

    def verificar_paths(self):
        """
        Verifica que existan los directorios de entrada y salida para
        la generación de consensos.

        - Verifica siempre la existencia del directorio de volúmenes ground truth.

        - Modo fold:
            * Verifica las rutas de todos los pacientes del fold.

        - Modo paciente individual:
            * Verifica únicamente las rutas del paciente especificado.
        """

        self._verificar_paths_gt()

        if self.es_fold:
            self._verificar_paths_fold()

        elif self.es_paciente_individual:
            self._verificar_paths_paciente()

    # =============================
    #        REPRESENTACIÓN
    # =============================

    def __repr__(self):
        """Representación de la instancia de ConfigConsenso."""
        if self.paciente is not None:
            return f"{self.__class__.__name__}(modelo={self.modelo.model_string}, epochs={self.epochs}, paciente={self.paciente.id})"
        else:
            return f"{self.__class__.__name__}(modelo={self.modelo.model_string}, epochs={self.epochs}, fold={self.fold_test})"
