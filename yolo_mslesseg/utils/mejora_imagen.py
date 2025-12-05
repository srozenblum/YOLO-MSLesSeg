"""
Script: mejora_imagen.py

Descripción:
    Define una estructura orientada a objetos para aplicar distintas técnicas de
    mejora de imagen 2D. Incluye una clase base 'Algoritmo' y cuatro implementaciones
    concretas: HE, CLAHE, GC y LT. Cada clase implementa su propio método 'aplicar',
    que ejecuta la técnica correspondiente sobre una imagen de entrada.
"""

import cv2
import numpy as np

from yolo_mslesseg.utils.utils import convertir_a_bgr


# =============================
#         CLASE BASE
# =============================


class Algoritmo:
    """
    Clase: Algoritmo

    Descripción:
        Clase base abstracta para técnicas de mejora de imagen 2D.
        Define la interfaz común que deben implementar las clases hijas.
    """

    def aplicar(self, imagen):
        """Método abstracto que debe implementarse en las clases hijas."""
        raise NotImplementedError(
            "El método aplicar debe ser implementado por la clase hija."
        )


# =============================
#     TÉCNICAS DE MEJORA
# =============================


class HE(Algoritmo):
    """
    Clase: HE

    Descripción:
        Implementa la Ecualización de Histograma (HE), mejorando el contraste global
        de la imagen mediante la redistribución uniforme de la intensidad.
    """

    def aplicar(self, imagen):
        """Aplicar HE sobre la imagen."""

        # Convertir la imagen a BGR si es RGB o gris
        img_bgr = convertir_a_bgr(imagen)

        # Convertir la imagen de BGR a YUV
        img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)

        # Ecualizar el canal de luminancia Y
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

        # Convertir de nuevo a RGB
        img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

        return img_rgb

    def __repr__(self):
        return "HE"


class CLAHE(Algoritmo):
    """
    Clase: CLAHE

    Descripción:
        Implementa la Ecualización Adaptativa de Histograma Limitado por Contraste (CLAHE),
        que ajusta el contraste de forma local evitando la sobreamplificación del ruido.

    Atributos:
        clip_limit (float): límite de contraste para la ecualización (por defecto 2.0).
        tile_grid_size (tuple[int, int]): tamaño de la cuadrícula para el procesamiento local (por defecto (8, 8)).
    """

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def aplicar(self, imagen):
        """Aplicar CLAHE sobre el canal L de la imagen."""

        # Convertir la imagen a BGR si es RGB o gris
        img_bgr = convertir_a_bgr(imagen)

        # Convertir de BGR a LAB
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

        # Separar los canales L, A, B
        l, a, b = cv2.split(lab)

        # Crear el objeto CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )

        # Aplicar CLAHE solo sobre el canal L (luminancia)
        l_clahe = clahe.apply(l)

        # Volver a unir los canales L (modificado), A, B
        img_merge = cv2.merge((l_clahe, a, b))

        # Convertir la imagen de nuevo de LAB a BGR
        image = cv2.cvtColor(img_merge, cv2.COLOR_LAB2BGR)

        return image

    def __repr__(self):
        return "CLAHE"


class GC(Algoritmo):
    """
    Clase: GC

    Descripción:
        Implementa la Corrección Gamma (GC), ajustando el brillo y contraste
        mediante una transformación no lineal de los valores de intensidad.

    Atributos:
        gamma (float): factor de corrección gamma (por defecto 2.0).
    """

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def aplicar(self, imagen):
        """Aplicar GC sobre la imagen."""

        # Convertir la imagen a BGR si es RGB o gris
        img_bgr = convertir_a_bgr(imagen)

        # Crear la tabla de corrección gamma
        table = np.array((np.linspace(0, 1, 256) ** self.gamma) * 255, dtype=np.uint8)

        # Aplicar la tabla de corrección gamma a la imagen
        img_rgb = cv2.LUT(img_bgr, table)

        return img_rgb

    def __repr__(self):
        return "GC"


class LT(Algoritmo):
    """
    Clase: LT

    Descripción:
        Implementa la Transformación Logarítmica (LT), que realza los detalles
        en regiones oscuras comprimiendo el rango dinámico de intensidades.
    """

    def aplicar(self, imagen):
        """Aplicar LT sobre la imagen."""

        # Convertir la imagen a BGR si es RGB o gris
        img_bgr = convertir_a_bgr(imagen)

        # Convertir a uint16 para evitar problemas con valores grandes
        img_bgr = img_bgr.astype(np.uint16)

        # Calcular el valor de c
        c = 255 / np.log(1 + img_bgr.max())

        # Aplicar la transformación logarítmica
        img_log = c * np.log(1 + img_bgr)

        # Convertir el resultado a uint8 para que sea compatible con OpenCV y la visualización
        img_rgb = np.clip(img_log, 0, 255).astype(np.uint8)

        return img_rgb

    def __repr__(self):
        return "LT"
