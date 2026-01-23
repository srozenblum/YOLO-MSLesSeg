import nibabel as nib
import numpy as np

from yolo_mslesseg.utils.mejora_imagen import HE, CLAHE, GC, LT
from yolo_mslesseg.utils.paths import DATASET_MSLESSEG
from yolo_mslesseg.utils.utils import ruta_existente


class Paciente:
    """
    Clase: Paciente

    Descripción:
        Representa un paciente del dataset MSLesSeg, con sus modalidades, timepoints
        (si existen), planos y máscaras asociadas. Permite acceder, mejorar y extraer
        cortes de los volúmenes de resonancia magnética (T1, T2, FLAIR) y del ground
        truth en distintos planos anatómicos.

        Se utiliza en las etapas de extracción de cortes, entrenamiento y evaluación,
        proporcionando un manejo centralizado de volúmenes, máscaras y algoritmos de mejora.

    Convención de directorios:
        MSLesSeg-Dataset/
            └── train/
                └── PX/
                    ├── T1/
                    │   ├── PX_T1_T1.nii.gz
                    │   ├── PX_T1_T2.nii.gz
                    │   ├── PX_T1_FLAIR.nii.gz
                    │   └── PX_T1_MASK.nii.gz
                    ├── T2/
                    └── ...

    Atributos:
        id (str):
            Identificador del paciente (formato 'PX').

        plano (str):
            Orientación anatómica ('axial', 'coronal', 'sagital') o 'consenso'.

        timepoint (str, opcional):
            Punto temporal de adquisición de imagen MRI. Por defecto 'T1'.

        modalidad (list[str], opcional):
            Modalidades de imagen MRI consideradas (T1, T2, FLAIR). Por defecto todas.

        modalidad_str (str):
            Representación concatenada de las modalidades de imagen (ej. 'T1T2FLAIR').

        mejora (str, opcional):
            Algoritmo de mejora de imagen a aplicar ('HE', 'CLAHE', 'GC', 'LT', o None).
            Por defecto None.

        base_dir (Path):
            Ruta base del paciente en el dataset MSLesSeg.

        gt_mask (np.ndarray):
            Volumen ground truth cargado en memoria.

        _volumenes (dict[str, np.ndarray]):
            Diccionario con los volúmenes cargados por modalidad.
    """

    MODALIDADES = ("T1", "T2", "FLAIR")
    MEJORAS = ("HE", "CLAHE", "GC", "LT")
    PLANOS = ("axial", "coronal", "sagital", "consenso")
    TIMEPOINTS = ("T1", "T2", "T3", "T4")

    def __init__(
        self, id, plano, timepoint="T1", modalidad=None, mejora=None, gt_mask=None
    ):
        # --- Validación de argumentos ---
        self._validar_argumentos(id, plano, timepoint, mejora, modalidad)

        # --- Atributos principales ---
        self._set_atributos_principales(
            id, plano, timepoint, modalidad, mejora, gt_mask
        )

    # ======================================
    #   MÉTODOS AUXILIARES DEL CONSTRUCTOR
    # ======================================

    def _validar_argumentos(self, id, plano, timepoint, mejora, modalidad):
        if not id.startswith("P"):
            raise ValueError(
                f"ID de paciente no válido: '{id}'. "
                "Debe seguir el formato 'P#' (por ejemplo: P1, P12, P53)."
            )

        if plano not in self.PLANOS:
            raise ValueError(f"Plano {plano} no válido.")

        if timepoint not in self.TIMEPOINTS:
            raise ValueError(f"Timepoint {timepoint} no válido.")

        if mejora is not None and mejora not in self.MEJORAS:
            raise ValueError(
                f"Algoritmo de mejora '{mejora}' no válido. Opciones: {self.MEJORAS}"
            )

        if not isinstance(modalidad, list) or not modalidad:
            raise TypeError(
                "Modalidad debe ser una lista no vacía (por ejemplo, ['T1', 'T2'] o ['T1','T2','FLAIR'])"
            )

        invalidas = [m for m in modalidad if m not in self.MODALIDADES]
        if invalidas:
            raise ValueError(f"Modalidades no reconocidas: {invalidas}")

    def _set_atributos_principales(
        self, id, plano, timepoint, modalidad, mejora, gt_mask
    ):
        self.id = id
        self.base_dir = DATASET_MSLESSEG / id  # Directorio base en el dataset
        self.plano = plano
        self.timepoint = timepoint
        self.sin_timepoints = not any(
            (self.base_dir / tp).exists() for tp in self.TIMEPOINTS
        )
        self.mejora = mejora
        self._gt_mask = gt_mask
        self._volumenes = {}  # Imagenes por modalidad

        # Normalizar modalidades y generar string
        self.modalidad = list(dict.fromkeys(modalidad))
        self.modalidad_str = "".join(
            [m for m in self.MODALIDADES if m in set(self.modalidad)]
        )

    # =============================
    #          RUTAS
    # =============================

    def volumen_path(self, modalidad):
        if self.sin_timepoints:
            return self.base_dir / f"{self.id}_{modalidad}.nii.gz"
        return (
            self.base_dir
            / self.timepoint
            / f"{self.id}_{self.timepoint}_{modalidad}.nii.gz"
        )

    @property
    def gt_mask_path(self):
        """Devuelve la ruta de la máscara ground truth."""
        if self.sin_timepoints:
            return self.base_dir / f"{self.id}_MASK.nii.gz"
        return (
            self.base_dir / self.timepoint / f"{self.id}_{self.timepoint}_MASK.nii.gz"
        )

    # =============================
    #         CARGA DE DATOS
    # =============================

    def cargar_volumen(self, modalidad):
        """
        Devuelve el volumen 3D de la modalidad indicada y
        lo guarda en el diccionario interno _volumen si no estaba cargado.
        """
        if modalidad not in self._volumenes:
            vol_path = self.volumen_path(modalidad)
            if not ruta_existente(vol_path):
                raise FileNotFoundError(f"No se encontró el volumen {modalidad}.")
            self._volumenes[modalidad] = nib.load(vol_path).get_fdata()
        return self._volumenes[modalidad]

    @property
    def gt_mask(self):
        """Devuelve la máscara binaria (ground truth)."""
        if self._gt_mask is None:
            if not ruta_existente(self.gt_mask_path):
                raise FileNotFoundError(
                    f"No se encontró la máscara en {self.gt_mask_path}"
                )
            self._gt_mask = nib.load(self.gt_mask_path).get_fdata()
        return self._gt_mask

    @property
    def num_cortes(self):
        """Devuelve el número total de cortes de la máscara según el plano."""

        mapping = {"axial": 2, "coronal": 1, "sagital": 0}
        if self.plano not in mapping:
            raise ValueError(f"Plano no reconocido: {self.plano}")
        return self.gt_mask.shape[mapping[self.plano]]

    # =============================
    #         PROCESAMIENTO
    # =============================

    def aplicar_mejora(self, imagen):
        """Aplica el algoritmo de mejora indicado, si corresponde."""
        if self.mejora is None:
            return imagen

        mejora_map = {
            "HE": HE,
            "CLAHE": CLAHE,
            "GC": GC,
            "LT": LT,
        }

        mejora_cls = mejora_map.get(self.mejora)
        if mejora_cls is None:
            raise ValueError(f"Mejora no reconocida: {self.mejora}.")
        return mejora_cls().aplicar(imagen)

    # =============================
    #         EXTRACCIÓN
    # =============================

    def obtener_corte_imagen(self, i, modalidad):
        """
        Extrae el corte i-ésimo del volumen según el plano del paciente y la modalidad
        y aplica el algoritmo de mejora correspondiente.
        """
        corte = self.cargar_volumen(modalidad)[self.indice_plano(i)]
        return self.aplicar_mejora(imagen=corte)

    def obtener_corte_mascara(self, i):
        """
        Extrae el corte i-ésimo de la máscara según el plano del paciente.
        """
        return self.gt_mask[self.indice_plano(i)]

    def indice_plano(self, i):
        """
        Devuelve una tupla que contiene los índices de slicing
        según el plano correspondiente.
        """
        if self.plano == "consenso":
            raise ValueError(
                "El plano 'consenso' no es un plano anatómico y no admite extracción de índices."
            )

        mapping = {
            "axial": (slice(None), slice(None), i),
            "coronal": (slice(None), i, slice(None)),
            "sagital": (i, slice(None), slice(None)),
        }

        return mapping[self.plano]

    # =============================
    #     DETECCIÓN DE LESIONES
    # =============================

    def indices_cortes_con_lesion(self):
        """Devuelve los índices de cortes que contienen lesión en la máscara."""
        indices = [
            i
            for i in range(self.num_cortes)
            if np.any(self.obtener_corte_mascara(i) > 0)
        ]
        return indices

    def indices_a_usar(self, num_cortes=None):
        """
        Devuelve los índices de cortes con lesión a usar.
        - Si num_cortes es None, o hay menos cortes que num_cortes, devuelve todos.
        - Si hay más cortes que num_cortes, devuelve los cortes centrales.
        """
        indices_validos = self.indices_cortes_con_lesion()
        if num_cortes is None or len(indices_validos) <= num_cortes:
            return indices_validos

        centro = len(indices_validos) // 2
        mitad = num_cortes // 2
        start = max(0, centro - mitad)
        end = start + num_cortes
        return indices_validos[start:end]

    # =============================
    #     EXTRACCIÓN POR LESIÓN
    # =============================

    def cortes_con_lesion_img(self, num_cortes=None):
        """
        Devuelve todos los cortes del volumen que contienen lesión,
        organizados por modalidad, en formato de diccionario. Cada clave es
        una modalidad (T1, T2, FLAIR) y el valor es
        una lista de tuplas (índice_en_volumen, corte):

        {"T1": [(i0, corte0), (i1, corte1), ...],
         "T2": [(i0, corte0), ...],
         "FLAIR": [...]}
        """
        cortes_dict = {}
        indices = self.indices_a_usar(num_cortes)
        for m in self.modalidad:
            lista_cortes = []
            for i in indices:
                corte = self.obtener_corte_imagen(i, modalidad=m)
                lista_cortes.append((i, corte))
            cortes_dict[m] = lista_cortes
        return cortes_dict

    def cortes_con_lesion_mask(self, num_cortes=None):
        """
        Devuelve todos los cortes de la máscara que contienen lesión,
        en formato de lista de tuplas [(indice, corte), ...].
        """
        indices = self.indices_a_usar(num_cortes)
        return [(i, self.obtener_corte_mascara(i)) for i in indices]

    def __repr__(self):
        """Representación interna de la instancia de Paciente."""
        return f"Paciente({self.id})"

    def __str__(self):
        """Representación legible de la instancia de Paciente."""
        return self.id
