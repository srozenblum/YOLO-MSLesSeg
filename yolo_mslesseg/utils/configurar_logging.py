"""
Script: configurar_logging.py

Descripción:
    Configura el sistema global de logging utilizado en todo el pipeline.
    Proporciona:
        - Niveles personalizados:
              * SKIP   → para indicar que un resultado ya existe (⏩).
              * HEADER → para encabezados claros por etapa (negrita).
        - Salida coloreada en consola mediante códigos ANSI.
        - Log limpio en archivo ('pipeline.log') sin códigos ANSI.
        - Función unificada get_logger() para obtener el logger de cada script.

Modo de uso:
    from configurar_logging import get_logger
    logger = get_logger(_file_)

Convenciones:
    - Toda la configuración se realiza una sola vez en este módulo.
    - Todos los scripts deben obtener su logger a través de get_logger().
    - El archivo pipeline.log se sobrescribe en cada nueva ejecución del pipeline.
"""

import logging
import re
import sys
from pathlib import Path

# ============================================================
#               NIVELES PERSONALIZADOS
# ============================================================


def registrar_nivel_personalizado(valor, nombre):
    """Registra un nivel de logging personalizado y añade logger.<nombre_en_minusculas>()."""
    logging.addLevelName(valor, nombre)

    def log_method(self, message, *args, **kwargs):
        if self.isEnabledFor(valor):
            self._log(valor, message, args, **kwargs)

    setattr(logging.Logger, nombre.lower(), log_method)
    return valor


# Niveles adicionales
SKIP_LEVEL = registrar_nivel_personalizado(23, "SKIP")  # ⏩ resultados ya existentes
HEADER_LEVEL = registrar_nivel_personalizado(35, "HEADER")  # Encabezados de etapa

# Expresión regular para eliminar códigos ANSI
ANSI_ESCAPE = re.compile(r"\x1B\[[0-?][ -/][@-~]")

# ============================================================
#                   FORMATTERS
# ============================================================


class ColorFormatter(logging.Formatter):
    """Formatter con colores ANSI para la consola."""

    COLORS = {
        logging.DEBUG: "\033[90m",  # Gris
        logging.INFO: "\033[38;5;39m",  # Azul brillante
        logging.WARNING: "\033[1;93m",  # Amarillo en negrita
        logging.ERROR: "\033[1;91m",  # Rojo en negrita
        logging.CRITICAL: "\033[1;97;41m",  # Blanco en negrita con fondo rojo
        SKIP_LEVEL: "\033[38;5;25m",  # Azul oscuro
        HEADER_LEVEL: "\033[1;97m",  # Blanco en negrita
    }

    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}"


class NoColorFormatter(logging.Formatter):
    """Formatter que elimina códigos ANSI antes de escribir al archivo."""

    def format(self, record):
        raw = super().format(record)
        return ANSI_ESCAPE.sub("", raw)


# ============================================================
#               CONFIGURACIÓN GLOBAL DE LOGGING
# ============================================================


def configurar_logging(level=logging.INFO, log_file=None):
    """
    Configura logging global:
        - Handler coloreado para consola con UTF-8.
        - Handler sin colores para archivo con UTF-8.
        - Niveles personalizados SKIP y HEADER habilitados.

    Esta función se llama automáticamente al cargar este módulo.
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    # 1. Console handler con colores y UTF-8
    # Reconfigurar sys.stdout para usar UTF-8 en Windows
    if sys.platform == "win32":
        # Intentar reconfigurar stdout con UTF-8
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            import codecs

            sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(ColorFormatter("%(message)s"))
    logger.addHandler(ch)

    # 2. File handler sin colores con UTF-8
    if log_file is not None:
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setFormatter(NoColorFormatter("%(message)s"))
        logger.addHandler(fh)

    return logger


# Se configura el logging al importar este módulo
configurar_logging()


def configurar_logging_demo():
    """
    Configura logging para la ejecución de la demo.
    """
    logger = logging.getLogger()

    # Eliminar SOLO el FileHandler que apunta a pipeline.log
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            # Evitar borrar otros handlers futuros por accidente
            if "pipeline.log" in str(getattr(h, "baseFilename", "")):
                logger.removeHandler(h)

    # demo.log relativo al cwd (demo/)
    demo_log_path = Path.cwd() / "demo.log"

    # Añadir FileHandler para demo.log con UTF-8
    demo_handler = logging.FileHandler(demo_log_path, mode="w", encoding="utf-8")
    demo_handler.setLevel(logging.INFO)
    demo_handler.setFormatter(NoColorFormatter("%(message)s"))
    logger.addHandler(demo_handler)


# ============================================================
#               FUNCIÓN PÚBLICA PARA SCRIPTS
# ============================================================


def get_logger(source_file):
    """
    Devuelve un logger específico para un script.

    Parámetro:
        source_file (str | Path): ruta del script (_file_ recomendado)

    Ejemplo:
        logger = get_logger(_file_)
    """
    name = Path(source_file).stem
    return logging.getLogger(name)
