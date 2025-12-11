"""
Script: analizar_pacientes_dsc.py

Descripción:
    Analiza los resultados individuales de cada paciente dentro de una configuración
    experimental concreta y, para cada algoritmo de mejora disponible, determina
    qué pacientes obtuvieron el mejor y el peor DSC, indicando además el plano en
    el que se obtuvo dicha métrica (sin tener en cuenta el consenso).

    Este script escanea el árbol:

    results/<mejora>/<modalidad>_<num_cortes>c_<k_folds>folds_<epochs>epochs/foldX/PX/PX_<plano>_results.json

    donde cada archivo corresponde a un paciente en un plano dado, y contiene métricas individuales
    generadas por `eval.py`.

Modo de ejecución:
    Este script debe ejecutarse únicamente por CLI. No es parte del pipeline, por lo que
    no está preparado para uso interno.

Argumentos CLI:
    --plano (str, requerido)
        Plano anatómico de extracción ('axial', 'coronal', 'sagital').
        Se usa solo para construir el objeto Modelo.

    --modalidad (list[str], opcional)
        Modalidad o modalidades de imagen MRI ('T1', 'T2', 'FLAIR').
        Por defecto todas. Se usa para localizar el experimento.

    --mejora (str, opcional)
        Algoritmo de mejora aplicado ('HE', 'CLAHE', 'GC', 'LT', o None).
        Por defecto None. No afecta al filtrado, solo se usa para localizar el experimento.

    --epochs (int, requerido)
        Número de épocas del modelo entrenado.

    --k_folds (int, opcional)
        Número de folds usados en validación cruzada.
        Por defecto 5.

Uso por CLI:
    python -m yolo_mslesseg.extras.analizar_pacientes_dsc \
        --plano axial \
        --modalidad FLAIR \
        --num_cortes P50 \
        --epochs 50

Entradas:
    - JSONs de pacientes: PX_<plano>_results.json

Salidas:
    - Impresión por consola: mejor y peor paciente para cada mejora.
"""

import argparse
from pathlib import Path

from yolo_mslesseg.utils.Modelo import Modelo
from yolo_mslesseg.utils.configurar_logging import get_logger
from yolo_mslesseg.utils.utils import (
    ruta_existente,
    leer_json,
    int_o_percentil,
    construir_nombre_configuracion,
)

logger = get_logger(__file__)

PLANOS_VALIDOS = ["axial", "coronal", "sagital"]

# =============================
#       FUNCIONES BASE
# =============================


def extraer_plano_desde_json(json_path):
    """
    Dado un archivo: PX_<plano>_results.json
    extrae el <plano>, ignorando cualquier otro plano no permitido.
    """
    stem = json_path.stem  # Ej: "P1_axial_results"

    # Omitir consenso
    if "consenso" in stem:
        return None

    for plano in PLANOS_VALIDOS:
        if (
            f"_{plano}_" in stem
            or stem.endswith(f"_{plano}")
            or stem.startswith(f"{plano}_")
        ):
            return plano

    return None


def extraer_dsc_paciente(json_path):
    """
    Extrae el DSC a partir de un JSON individual.
    """
    try:
        data = leer_json(json_path)
    except Exception as e:
        logger.warning(f"⚠️ No se pudo leer {json_path}: {e}")
        return None

    return data.get("DSC", None)


def imprimir_resultados(dsc_mejoras):
    """
    Imprime los resultados del análisis:
    mejor y peor paciente por mejora (con su plano).
    """
    for mejora, pacientes_info in dsc_mejoras.items():
        if not pacientes_info:
            logger.info(f"ℹ️ Mejora {mejora} sin datos.")
            continue

        # mejor / peor por DSC
        mejor = max(pacientes_info, key=lambda px: pacientes_info[px]["dsc"])
        peor = min(pacientes_info, key=lambda px: pacientes_info[px]["dsc"])

        print(f"\n===== {mejora} =====")
        print(
            f"  Mejor paciente: {mejor} "
            f"(DSC = {pacientes_info[mejor]['dsc']:.4f}, "
            f"plano = {pacientes_info[mejor]['plano']})"
        )
        print(
            f"  Peor paciente:  {peor}  "
            f"(DSC = {pacientes_info[peor]['dsc']:.4f}, "
            f"plano = {pacientes_info[peor]['plano']})"
        )


def analizar_experimento(results_dir, configuracion_global):
    """
    Recorre results/<mejora>/<config>/foldX/PX/*.json
    y construye:
        mejora → paciente → { dsc, plano }
    """
    if not ruta_existente(results_dir):
        logger.error(f"❌ No existe el directorio results/: {results_dir}")
        return

    dsc_por_mejora = {}

    # Recorrer mejoras
    for mejora_dir in results_dir.iterdir():
        if not mejora_dir.is_dir():
            continue

        mejora = mejora_dir.name.upper()
        experimento_dir = mejora_dir / configuracion_global

        if not experimento_dir.exists():
            continue

        dsc_por_mejora.setdefault(mejora, {})

        # Recorrer folds
        for fold_dir in experimento_dir.iterdir():
            if not fold_dir.is_dir() or not fold_dir.name.startswith("fold"):
                continue

            # Pacientes
            for px_dir in fold_dir.iterdir():
                if not px_dir.is_dir():
                    continue

                paciente_id = px_dir.name

                # JSONs individuales del paciente
                for json_file in px_dir.glob("*.json"):
                    plano = extraer_plano_desde_json(json_file)
                    if plano not in PLANOS_VALIDOS:
                        continue  # Ignorar consenso o desconocidos

                    dsc = extraer_dsc_paciente(json_file)
                    if dsc is None:
                        continue

                    # Guardar si es mejor que lo previo
                    previo = dsc_por_mejora[mejora].get(paciente_id, None)
                    if previo is None or dsc > previo["dsc"]:
                        dsc_por_mejora[mejora][paciente_id] = {
                            "dsc": dsc,
                            "plano": plano,
                        }

    if not dsc_por_mejora:
        logger.warning("⚠️ No se encontraron métricas individuales.")
        return

    imprimir_resultados(dsc_por_mejora)


# =============================
#        FLUJO PRINCIPAL
# =============================


def ejecutar_flujo(modelo, epochs):
    """Ejecuta el flujo de análisis completo."""
    configuracion_global = construir_nombre_configuracion(modelo, epochs)

    results_dir = Path("results")

    logger.header(f"🔍 Analizando pacientes para {configuracion_global}")

    analizar_experimento(
        results_dir=results_dir, configuracion_global=configuracion_global
    )


# =============================
#       CLI Y EJECUCIÓN
# =============================


def parsear_args():
    """Parseo de argumentos CLI."""
    parser = argparse.ArgumentParser(
        description="Analizar mejores y peores pacientes por mejora según DSC."
    )
    parser.add_argument(
        "--plano",
        type=str,
        required=True,
        choices=["axial", "coronal", "sagital"],
        metavar="[axial, coronal, sagital]",
        help="Plano anatómico del modelo.",
    )
    parser.add_argument(
        "--modalidad",
        nargs="+",
        choices=["T1", "T2", "FLAIR"],
        default=["T1", "T2", "FLAIR"],
        metavar="[T1, T2, FLAIR]",
        help="Modalidad(es) de imagen MRI. Por defecto todas.",
    )
    parser.add_argument(
        "--num_cortes",
        type=int_o_percentil,
        required=True,
        metavar="<num_cortes>",
        help="Número de cortes extraídos (valor fijo o percentil).",
    )
    parser.add_argument(
        "--mejora",
        type=str,
        default=None,
        choices=["HE", "CLAHE", "GC", "LT"],
        metavar="<mejora>",
        help="Algoritmo de mejora de imagen aplicado. Por defecto None.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        metavar="<epochs>",
        help="Número de épocas de entrenamiento.",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        metavar="<k_folds>",
        help="Número de folds para validación cruzada. Por defecto 5.",
    )
    return parser.parse_args()


def main():
    """
    Entrada CLI del script: parsea argumentos, construye Modelo
    y ejecuta el flujo completo.
    """
    args = parsear_args()

    modelo = Modelo(
        plano=args.plano,
        num_cortes=args.num_cortes,
        modalidad=args.modalidad,
        k_folds=args.k_folds,
        mejora=args.mejora,
    )

    ejecutar_flujo(modelo=modelo, epochs=args.epochs)


if __name__ == "__main__":
    main()
