from pathlib import Path


class Modelo:
    """
    Clase: Modelo

    Descripción:
        Representa la configuración estructural de un modelo YOLO empleado en el
        pipeline YOLO-MSLesSeg. Gestiona la identificación y nomenclatura del
        modelo según el plano anatómico, las modalidades empleadas, el número de
        cortes utilizados, el número de folds para validación cruzada y el tipo
        de técnica mejora de imagen aplicada.

        Se utiliza en todas las etapas del pipeline, asegurando una nomenclatura
        consistente entre los distintos experimentos del flujo de trabajo.

    Atributos:
        plano (str):
            Plano anatómico empleado ('axial', 'coronal', 'sagital') o 'consenso'.

        num_cortes (int_o_percentil):
            Número de cortes usados (valor entero o percentil 'PX').

        modalidad (list[str]):
            Modalidad o modalidades de imagen MRI consideradas (T1, T2, FLAIR).

        modalidad_str (str):
            Representación concatenada de las modalidades de imagen (ej. 'T1T2FLAIR').

        k_folds (int):
            Número de folds usados para validación cruzada.

        mejora (str, opcional):
            Algoritmo de mejora aplicado ('HE', 'CLAHE', 'GC', 'LT', o None).
            Por defecto None.
    """

    PLANOS = ("axial", "coronal", "sagital", "consenso")
    MEJORAS = (None, "HE", "CLAHE", "GC", "LT")

    def __init__(self, plano, num_cortes, modalidad, k_folds, mejora=None):
        # --- Validación de argumentos ---
        self._validar_argumentos(plano, mejora)

        # --- Atributos principales ---
        self._set_atributos_principales(
            plano,
            num_cortes,
            modalidad,
            k_folds,
            mejora,
        )

    # ======================================
    #   MÉTODOS AUXILIARES DEL CONSTRUCTOR
    # ======================================

    def _validar_argumentos(self, plano, mejora):
        if plano not in self.PLANOS:
            raise ValueError(
                f"Plano '{plano}' no válido. Debe ser uno de {self.PLANOS}."
            )
        if mejora not in self.MEJORAS:
            raise ValueError(
                f"Mejora '{mejora}' no válida. Debe ser uno de {self.MEJORAS}."
            )

    def _set_atributos_principales(self, plano, num_cortes, modalidad, k_folds, mejora):
        self.plano = plano.lower()
        self.num_cortes = num_cortes
        self.modalidad = modalidad
        self.modalidad_str = "".join(modalidad)
        self.k_folds = k_folds
        self.mejora = mejora.upper() if mejora else None

    # =============================
    #       IDENTIFICADORES
    # =============================

    @property
    def exp_string(self):
        """Nombre corto del experimento (Control o tipo de mejora)."""
        return self.mejora if self.mejora else "Control"

    @property
    def base_path(self):
        """Ruta base del modelo."""
        return (
            Path(self.exp_string)
            / f"{self.modalidad_str}_{self.num_cortes}c_{self.k_folds}folds"
        )

    @property
    def model_string(self):
        """Identificador único y legible del modelo según plano, modalidad y número de cortes."""
        if not self.mejora:
            return f"{self.plano}_{self.modalidad_str}_{self.num_cortes}c_{self.k_folds}folds"
        else:
            return f"{self.plano}_{self.modalidad_str}_{self.mejora}_{self.num_cortes}c_{self.k_folds}folds"

    # =============================
    #        REPRESENTACIÓN
    # =============================

    def __repr__(self):
        """Representación interna de la instancia de Modelo."""
        return f"Modelo({self.model_string})"

    def __str__(self):
        """Representación legible de la instancia de Modelo."""
        return f"{self.model_string}"
