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


class ConfigEval:
    """
    Clase: ConfigEval

    Descripción:
        Configuración y gestión de rutas/variables globales para las etapas de evaluación,
        implementadas en `eval.py` (métricas por paciente o por fold) y `promediar_folds.py`
        (métricas globales de experimento). Se encarga de verificar, crear y limpiar las
        estructuras de directorios tanto para un paciente individual como para un fold,
        asegurando la correcta localización de los volúmenes predichos y los de ground truth
        para el cálculo de métricas.

    Casos de uso contemplados:
        1. Modo paciente individual (`paciente` ≠ None)
           Calcula métricas solo para el paciente especificado.

        2. Modo fold (`paciente` = None, `fold_test` ≠ None)
           Calcula métricas para todos los pacientes del fold indicado.

        3. Modo experimento (`paciente` = None y `fold_test` = None)
           Calcula métricas globales del experimento (promedio de folds).
           Requiere que existan los resultados individuales de cada fold.

    Convenciones de directorios:
        results/: métricas de evaluación
        └── <mejora>/
             └── <modalidad>_<num_cortes>c_<k_folds>folds_<epochs>epochs/
                 ├── global_<plano>_results.json
                 ├── <fold_test>/
                 │   ├── <fold_test>_<plano>_results.json
                 │   ├── PX/
                 │   │   └── PX_<plano>_results.json
                 │   └── ...
                 └── ...

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
            Plano anatómico de procesamiento ('axial', 'coronal', 'sagital').
            Si se indica `plano_forzado`, reemplaza el plano original del modelo.

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
            Directorio base de los volúmenes reconstruidos del experimento.

        vols_fold_dir (Path):
            Directorio de los volúmenes reconstruidos del fold.

        gt_dir (Path):
            Directorio base de los volúmenes ground truth.

        results_fold_json (Path):
            Ruta al archivo JSON de las métricas del fold.

        results_experimento_json (Path):
            Ruta al archivo JSON de las métricas globales del experimento.

         # --- Atributos válidos solo en modo paciente ---
        paciente_vol_root (Path):
            Directorio base de los volúmenes reconstruidos del paciente.

        paciente_results_root (Path):
            Directorio base de las métricas del paciente.

        paciente_pred_vol (Path):
            Ruta al archivo NIfTI del volumen reconstruido del paciente
            en el plano del modelo.

        paciente_gt_vol (Path):
            Ruta al archivo NIfTI del ground truth del paciente.

        paciente_results_json (Path):
            Ruta al archivo JSON de las métricas del paciente.
    """

    def __init__(
        self,
        modelo,
        epochs,
        k_folds=5,
        paciente=None,
        fold_test=None,
        plano_forzado=None,
    ):
        # --- Atributos principales ---
        self._set_atributos_principales(
            modelo=modelo,
            epochs=epochs,
            k_folds=k_folds,
            paciente=paciente,
            fold_test=fold_test,
            plano_forzado=plano_forzado,
        )

        # --- Determinar modo de ejecución ---
        self._resolver_modo_ejecucion()

        # --- Directorios de volúmenes (entrada) ---
        self._resolver_rutas_vols()

        # --- Directorios de resultados (salida) ---
        self._resolver_rutas_resultados()

        # --- Rutas específicas del paciente (si aplica) ---
        self._resolver_rutas_paciente()

    # ======================================
    #   MÉTODOS AUXILIARES DEL CONSTRUCTOR
    # ======================================

    def _set_atributos_principales(
        self,
        modelo,
        epochs,
        k_folds,
        paciente,
        fold_test,
        plano_forzado,
    ):
        self.modelo = modelo
        self.plano = plano_forzado if plano_forzado is not None else modelo.plano
        self.epochs = epochs
        self.k_folds = k_folds
        self.paciente = paciente
        self.fold_test = fold_test

        # Directorio GT
        self.gt_dir = Path("GT") / "train"

    def _resolver_modo_ejecucion(self):
        self.es_paciente_individual = self.paciente is not None
        self.es_fold = not self.es_paciente_individual and self.fold_test is not None
        self.es_experimento = not self.es_paciente_individual and self.fold_test is None

        if self.es_paciente_individual:
            # Determinar fold asignado al paciente
            self.fold_test = calcular_fold(
                paciente_id=self.paciente.id,
                k_folds=self.k_folds,
            )
            return

        # En modo fold y modo experimento no hay nada más que resolver

    def _resolver_rutas_vols(self):
        self.vols_base_dir = (
            Path("vols") / f"{self.modelo.base_path}_{self.epochs}epochs"
        )

        # Volúmenes del fold
        self.vols_fold_dir = self.vols_base_dir / f"fold{self.fold_test}"

    def _resolver_rutas_resultados(self):
        self.results_base_dir = (
            Path("results") / f"{self.modelo.base_path}_{self.epochs}epochs"
        )

        # Resultados del fold
        self.results_fold_dir = self.results_base_dir / f"fold{self.fold_test}"

        # JSON de métricas del fold
        self.results_fold_json = (
            self.results_fold_dir / f"fold{self.fold_test}_{self.plano}_results.json"
        )

        # JSON de métricas globales del experimento (promedio de folds)
        self.results_experimento_json = (
            self.results_base_dir / f"global_{self.plano}_results.json"
        )

    def _resolver_rutas_paciente(self):
        if not self.es_paciente_individual:
            return

        self.paciente_vol_root = (
            self.vols_base_dir / f"fold{self.fold_test}" / self.paciente.id
        )
        self.paciente_results_root = (
            self.results_base_dir / f"fold{self.fold_test}" / self.paciente.id
        )

        # Archivos NIfTI (volumen predicho y ground truth)
        self.paciente_pred_vol = (
            self.paciente_vol_root / f"{self.paciente.id}_{self.plano}.nii.gz"
        )
        self.paciente_gt_vol = (
            self.gt_dir / self.paciente.id / f"{self.paciente.id}_MASK.nii.gz"
        )

        # JSON de métricas del paciente
        self.paciente_results_json = (
            self.paciente_results_root / f"{self.paciente.id}_{self.plano}_results.json"
        )

    # =============================
    #          LIMPIEZA
    # =============================

    def _limpiar_resultados_fold(self):
        """
        Limpia el JSON de métricas del fold
        y los JSON individuales de todos los pacientes.
        """
        # JSON de métricas del fold
        if ruta_existente(self.results_fold_json):
            try:
                self.results_fold_json.unlink()
            except Exception as e:
                logger.warning(f"⚠️ No se pudo eliminar {self.results_fold_json}: {e}")

        if ruta_existente(self.results_fold_dir):
            pacientes = listar_pacientes(self.results_fold_dir)

            # JSON individuales por paciente
            for paciente_id in pacientes:
                paciente_dir = self.results_fold_dir / paciente_id
                if not paciente_dir.is_dir():
                    continue

                for archivo in paciente_dir.iterdir():
                    if (
                        archivo.is_file()
                        and archivo.suffix.lower() == ".json"
                        and self.plano.lower() in archivo.name.lower()
                    ):
                        try:
                            archivo.unlink()
                        except Exception as e:
                            logger.warning(f"⚠️ No se pudo eliminar {archivo}: {e}")

    def _limpiar_resultados_experimento(self):
        """
        Limpia el JSON de métricas globales del experimento.
        """
        if ruta_existente(self.results_experimento_json):
            try:
                self.results_experimento_json.unlink()
            except Exception as e:
                logger.warning(
                    f"⚠️ No se pudo eliminar {self.results_experimento_json}: {e}"
                )

    def limpiar_resultados(self):
        """
        Limpia los JSON de métricas según el plano del modelo
        y el modo de ejecución.

        - Modo fold:
          Limpia los JSON de métricas de todos los pacientes del fold
          y el JSON del fold.

        - Modo paciente individual:
          Limpia únicamente el JSON de métricas del paciente especificado,
          sin afectar al resto del fold.

        - Modo experimento:
          Limpia únicamente el JSON de métricas globales del experimento.

        """
        if self.es_fold:
            self._limpiar_resultados_fold()

        elif self.es_paciente_individual:
            if ruta_existente(self.paciente_results_json):
                self.paciente_results_json.unlink()

        else:  # es_experimento
            self._limpiar_resultados_experimento()

    # =============================
    #         VERIFICACIÓN
    # =============================

    def _verificar_paths_fold(self):
        """
        Verifica que existan los archivos de entrada y el directorio de salida
        para los pacientes del fold.
        - Entrada: ground truth y predicciones por paciente (gt_dir y pred_vol_dir)
        - Salida: directorio raíz de métricas por paciente (results_root).
        """
        pacientes = listar_pacientes(self.vols_fold_dir)

        for paciente_id in pacientes:
            paciente_gt_dir = self.gt_dir / paciente_id / f"{paciente_id}_MASK.nii.gz"
            paciente_pred_vol_dir = (
                self.vols_fold_dir / paciente_id / f"{paciente_id}_{self.plano}.nii.gz"
            )
            results_root_paciente = self.results_fold_dir / paciente_id

            # gt_dir
            if not ruta_existente(paciente_gt_dir):  # Lanza excepción si no existe
                raise FileNotFoundError(
                    f"No existe el volumen ground truth del paciente {paciente_id}: {paciente_gt_dir}."
                )

            # pred_vol_dir
            if not ruta_existente(
                paciente_pred_vol_dir
            ):  # Lanza excepción si no existe
                raise FileNotFoundError(
                    f"No existe la predicción del paciente {paciente_id}: {paciente_pred_vol_dir}. "
                )

            # results_root
            crear_directorio(results_root_paciente)  # Asegurar salida

    def _verificar_paths_paciente(self):
        """
        Verifica que existan los directorios de entrada y salida para un paciente individual.
        - Entrada: GT y volumen predicho (paciente_gt_vol y paciente_pred_vol).
        - Salida: directorio raíz de métricas (paciente_results_root).
        """
        # paciente_gt_vol
        if not ruta_existente(self.paciente_gt_vol):  # Lanza excepción si no existe
            raise FileNotFoundError(
                f"No existe la GT del paciente {self.paciente.id}: {self.paciente_gt_vol}."
            )

        # paciente_pred_vol
        if not ruta_existente(self.paciente_pred_vol):  # Lanza excepción si no existe
            raise FileNotFoundError(
                f"No existe la predicción del paciente {self.paciente.id}: {self.paciente_pred_vol}."
            )

        # paciente_results_root
        crear_directorio(self.paciente_results_root)  # Asegurar salida

    def _verificar_paths_experimento(self):
        """
        Verifica que existan los JSON de métricas de cada fold
        para poder calcular el promedio a nivel de experimento.
        """
        if not ruta_existente(self.results_base_dir):
            raise FileNotFoundError(
                f"No existe el directorio de resultados: {self.results_base_dir}"
            )

        # Archivos de entrada esperados → `k_folds` archivos de resultados individuales de cada fold
        folds_esperados = [f"fold{i}_{self.plano}" for i in range(1, self.k_folds + 1)]
        folds_encontrados = set()

        # Buscar en todas las subcarpetas del experimento
        for fold_dir in self.results_base_dir.iterdir():
            if not fold_dir.is_dir() or not fold_dir.name.startswith("fold"):
                continue
            for archivo in fold_dir.iterdir():
                if archivo.is_file() and archivo.suffix.lower() == ".json":
                    for esperado in folds_esperados:
                        if archivo.name.startswith(esperado):
                            folds_encontrados.add(esperado)

        faltantes = [f for f in folds_esperados if f not in folds_encontrados]

        if faltantes:  # Lanza excepción si falta alguno
            raise FileNotFoundError(
                f"❌ No existen los JSON de resultados para los siguientes folds: {faltantes}"
            )

    def verificar_paths(self):
        """
        Verifica que existan los directorios de entrada y salida para
        el cálculo de métricas.

        - Modo fold:
            * Verifica las rutas de todos los pacientes del fold.

        - Modo paciente individual:
            * Verifica únicamente las rutas del paciente especificado.

        - Modo experimento:
            * Verifica las rutas de cada fold para poder calcular el promedio.
        """

        if self.es_fold:
            self._verificar_paths_fold()

        elif self.es_paciente_individual:
            self._verificar_paths_paciente()

        else:  # es_experimento
            self._verificar_paths_experimento()

    def __repr__(self):
        """Representación de la instancia de ConfigEval."""
        if self.paciente is not None:
            return f"{self.__class__.__name__}(modelo={self.modelo.model_string}, epochs={self.epochs}, paciente={self.paciente.id})"
        else:
            return f"{self.__class__.__name__}(modelo={self.modelo.model_string}, epochs={self.epochs}, fold={self.fold_test})"
