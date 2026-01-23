from pathlib import Path

from yolo_mslesseg.utils.configurar_logging import get_logger
from yolo_mslesseg.utils.paths import DATASET_MSLESSEG
from yolo_mslesseg.utils.utils import (
    ruta_existente,
    eliminar_directorio,
    crear_directorio,
    listar_pacientes,
    calcular_fold,
)

# Configurar logger
logger = get_logger(__file__)


class ConfigDataset:
    """
    Clase: ConfigDataset

    Descripción:
        Configuración y gestión de rutas/variables globales para la inicialización del
        dataset YOLO, implementada en `extraer_dataset.py`. Se encarga de verificar, crear
        y limpiar las estructuras de directorios tanto para un paciente individual como
        para el conjunto de pacientes completo, asegurando la correcta organización de
        imágenes, máscaras y anotaciones.

    Casos de uso contemplados:
        1. Modo paciente individual (`paciente` ≠ None):
           Construye la estructura de directorios solo para el paciente especificado.

        2. Modo completo (`paciente` = None, `completo` = True):
           Construye la estructura de directorios de todos los pacientes.

    Convenciones de directorios:
        MSLesSeg-Dataset/: dataset original de entrada

        datasets/: datasets YOLO con divisiones por fold y pacientes
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

    Atributos:
        modelo (Modelo):
            Instancia del modelo, que define plano, modalidades, mejora y base_path.

        plano (str):
            Plano anatómico de procesamiento ('axial', 'coronal' o 'sagital').
            Coincide con `modelo.plano`.

        dataset_entrada (Path):
            Directorio del dataset de entrada. Por defecto MSLesSeg-Dataset/train.

        k_folds (int):
            Número de folds para validación cruzada.

        completo (bool):
            Indica si se va a procesar el conjunto completo de pacientes.

        paciente (Paciente, opcional):
            Instancia del paciente si la ejecución es individual.
            Es None en ejecución completa.

        output_dir (Path):
            Directorio base de salida: datasets/<experimento>/<base_path>.

        # --- Atributos válidos solo en modo paciente ---
        fold_paciente (int, opcional):
            Número de fold al que pertenece el paciente.

        paciente_root (Path, opcional):
            Directorio base del paciente en el dataset YOLO.

        paciente_dir (dict[str, Path], opcional):
            Diccionario de subdirectorios del paciente:
                - images/
                - GT_masks/
                - labels/
    """

    def __init__(
        self,
        modelo,
        dataset_entrada=None,
        k_folds=5,
        completo=False,
        paciente=None,
    ):
        # --- Atributos principales ---
        self._set_atributos_principales(
            modelo, dataset_entrada, k_folds, completo, paciente
        )

        # --- Determinar modo de ejecución ---
        self._resolver_modo_ejecucion()

        # --- Rutas específicas del paciente (si aplica) ---
        self._resolver_rutas_paciente()

    # ======================================
    #  MÉTODOS AUXILIARES DEL CONSTRUCTOR
    # ======================================

    def _set_atributos_principales(
        self, modelo, dataset_entrada, k_folds, completo, paciente
    ):
        self.modelo = modelo
        self.plano = modelo.plano
        self.k_folds = k_folds
        self.paciente = paciente
        self.completo = completo

        # Dataset de entrada
        if dataset_entrada is None:
            self.dataset_entrada = DATASET_MSLESSEG
        else:
            self.dataset_entrada = Path(dataset_entrada)

        # Directorio de salida
        self.output_dir = Path("datasets") / f"{self.modelo.base_path}"

    def _resolver_modo_ejecucion(self):
        self.es_paciente_individual = self.paciente is not None
        self.es_completo = not self.es_paciente_individual and self.completo

        if self.es_paciente_individual:
            # Determinar fold del paciente
            self.fold_paciente = calcular_fold(
                paciente_id=self.paciente.id, k_folds=self.k_folds
            )

        elif self.es_completo:
            pass  # Nada más que hacer

        else:
            raise ValueError(
                "Debe especificarse un modo de ejecución: dataset completo o paciente individual."
            )

    def _resolver_rutas_paciente(self):
        if not self.es_paciente_individual:
            return

        self.paciente_root = (
            self.output_dir
            / f"fold{self.fold_paciente}"
            / self.paciente.id
            / self.plano
        )

        # Diccionario con los subdirectorios
        self.paciente_dir = {
            subdir: self.paciente_root / subdir
            for subdir in ["images", "GT_masks", "labels"]
        }

    # =============================
    #           LIMPIEZA
    # =============================

    def _limpiar_pacientes_fold(self, fold_dir):
        """
        Limpia las carpetas images/, GT_masks/ y labels/ en el plano
        correspondiente para todos los pacientes de un fold.
        """
        if ruta_existente(fold_dir):
            pacientes = listar_pacientes(fold_dir)

            for paciente_id in pacientes:
                paciente_path = fold_dir / paciente_id
                if not paciente_path.is_dir():
                    continue

                # Buscar carpeta del plano (axial, coronal, sagital)
                for plano_dir in paciente_path.iterdir():
                    if not plano_dir.is_dir():
                        continue
                    if self.plano.lower() not in plano_dir.name.lower():
                        continue

                    # Limpiar subdirectorios
                    for subdir in plano_dir.iterdir():
                        if subdir.is_dir():
                            try:
                                eliminar_directorio(subdir)
                            except Exception as e:
                                logger.warning(f"⚠️ No se pudo eliminar {subdir}: {e}")

    def _limpiar_dataset_completo(self):
        """
        Limpia los subdirectorios de los pacientes en el
        plano correspondiente para todos los folds.
        """
        if ruta_existente(self.output_dir):

            # Limpiar folds
            for fold_dir in self.output_dir.iterdir():
                if fold_dir.is_dir() and fold_dir.name.lower().startswith("fold"):
                    self._limpiar_pacientes_fold(fold_dir)

    def _limpiar_dataset_paciente(self):
        """
        Limpia los subdirectorios del plano correspondiente
        para un paciente individual.
        """
        if ruta_existente(self.paciente_root):

            # Iterar sobre los subdirectorios del paciente (images, GT_masks, labels)
            for nombre, subdir_path in self.paciente_dir.items():
                if not ruta_existente(subdir_path):
                    continue

                try:
                    eliminar_directorio(subdir_path)
                except Exception as e:
                    logger.warning(
                        f"⚠️ No se pudo eliminar {nombre} en {self.paciente.id}: {e}"
                    )

    def limpiar_dataset(self):
        """
        Limpia los archivos y carpetas en el directorio de salida
        según el plano y el modo de ejecución.

        - Modo dataset completo:
          Limpia todos los subdirectorios de todos
          los pacientes de todos los folds.

        - Modo paciente individual:
          Limpia todos los subdirectorios del paciente especificado,
          sin afectar al resto del fold.
        """
        if self.es_completo:
            self._limpiar_dataset_completo()

        else:  # es_paciente_individual
            self._limpiar_dataset_paciente()

    # =============================
    #         VERIFICACIÓN
    # =============================

    def _crear_estructura_folds(self, pacientes):
        """Crea la estructura de directorios para todos los pacientes del experimento."""
        # Crear directorios de folds
        for i in range(1, self.k_folds + 1):
            crear_directorio(self.output_dir / f"fold{i}")

        # Crear subestructura por paciente
        for paciente_id in pacientes:

            fold_paciente = calcular_fold(
                paciente_id=paciente_id,
                k_folds=self.k_folds,
            )

            paciente_dir = (
                self.output_dir / f"fold{fold_paciente}" / paciente_id / self.plano
            )
            crear_directorio(paciente_dir)

            for subdir in ["images", "GT_masks", "labels"]:
                crear_directorio(paciente_dir / subdir)

    def _verificar_paths_completo(self):
        """
        Verifica que existan los archivos de entrada y salida para el dataset completo.
        - Entrada: dataset original (self.dataset).
        - Salida: directorios por paciente en el dataset de salida.
        """
        # self.dataset_entrada
        if not self.dataset_entrada.is_dir():  # Lanza excepción si no existe
            raise FileNotFoundError(
                f"No existe el dataset de entrada: {self.dataset_entrada}"
            )

        # Directorio raíz
        crear_directorio(self.output_dir)

        # Estructura de directorios para folds
        pacientes = listar_pacientes(self.dataset_entrada)
        self._crear_estructura_folds(pacientes)  # Asegurar salida

    def _verificar_paths_paciente(self):
        """
        Verifica que existan los directorios de entrada y salida para un paciente individual.
        - Entrada: directorio del paciente en el dataset de entrada (paciente_input_dir).
        - Salida: directorio del paciente en el dataset de salida (subdirs en paciente_dir).
        """
        # paciente_input_dir
        paciente_input_dir = self.dataset_entrada / self.paciente.id
        if not paciente_input_dir.is_dir():  # Lanza excepción si no existe
            raise FileNotFoundError(
                f"No existe el directorio de entrada del paciente {self.paciente.id}: {paciente_input_dir}"
            )

        # subdirs
        for subdir in self.paciente_dir.values():
            crear_directorio(subdir)  # Asegurar salida

    def verificar_paths(self):
        """
        Verifica que existan los directorios de entrada y salida para
        la extracción del dataset.

        - Modo dataset completo:
            * Verifica la existencia del dataset de entrada.
            * Crea la estructura de folds según `k_folds`.

        - Modo paciente individual:
            * Verifica que exista el directorio de entrada del paciente especificado.
            * Crea los subdirectorios necesarios para ese paciente
              (images/, GT_masks/, labels/) dentro de su fold.
        """
        if self.es_completo:
            self._verificar_paths_completo()

        else:  # es_paciente_individual
            self._verificar_paths_paciente()

    # =============================
    #        REPRESENTACIÓN
    # =============================

    def __repr__(self):
        """Representación de la instancia de ConfigDataset."""
        if self.es_completo:
            return f"{self.__class__.__name__}(modelo={self.modelo.model_string}, completo={self.completo})"
        else:
            return f"{self.__class__.__name__}(modelo={self.modelo.model_string}, paciente={self.paciente.id})"
