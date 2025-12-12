"""
Script: ejecutar_pipeline.py

Descripción:
    Integra el flujo de trabajo completo, ya sea para un paciente individual
    o para un experimento completo. Ejecuta todas las etapas del pipeline de
    forma secuencial y controlada, integrando los módulos individuales de setup,
    extracción de dataset, entrenamiento, predicción, reconstrucción, consenso,
    evaluación y promedio de folds. Cada etapa detecta automáticamente si los
    resultados ya existen, evitando recomputar trabajo innecesario. Esto permite
    reiniciar ejecuciones sin perder progreso previo.

    Además, la etapa de entrenamiento es opcional. Por defecto, el pipeline
    no vuelve a entrenar los modelos YOLO en cada ejecución: esto evita
    un costo computacional innecesario y favorece la reproducibilidad
    cuando ya existen pesos entrenados. El entrenamiento se puede activar
    explícitamente en la llamada CLI.

Etapas:
    (0) Set up                    → descarga del dataset de entrada MSLesSeg
                                    y creación de la estructura de directorios.
    (1) Extraer dataset           → extracción del dataset YOLO y anotaciones.
    (2) Train (opcional)          → entrenamiento del modelo YOLO.
    (3) Generar predicciones      → predicción de máscaras de segmentación 2D.
    (4) Reconstruir volúmenes     → reconstrucción de volúmenes predichos.
    (5) Eval                      → cálculo de métricas de rendimiento.
    (6) Generar consensos         → generación de volúmenes consenso.

    # Si la ejecución es completa:
        (7) Promedio de folds  → cálculo de métricas globales

Argumentos CLI:
    --plano (str, requerido)
        Plano anatómico de extracción ('axial', 'coronal', 'sagital').

    --modalidad (list[str], opcional)
        Modalidad o modalidades de imagen MRI ('T1', 'T2', 'FLAIR').
        Por defecto todas.

    --num_cortes (int_o_percentil, requerido)
        Número de cortes a extraer (valor entero o percentil, por ejemplo 50 o 'P75').

    --mejora (str, opcional)
        Algoritmo de mejora de imagen ('HE', 'CLAHE', 'GC', 'LT', o None).
        Por defecto None.

    --k_folds (int, opcional)
        Número de folds para validación cruzada.
        Por defecto 5.

    --epochs (int, requerido)
        Número de épocas de entrenamiento.

    --umbral_consenso (int, opcional)
        Umbral de votación para el consenso (2 = mayoría simple, 3 = unanimidad).
        Por defecto 2.

    --completo (flag, excluyente con --paciente)
        Ejecutar el flujo de trabajo completo.

    --paciente_id (str, excluyente con --completo)
        Ejecutar el flujo de trabajo únicamente para el paciente indicado (ejemplo: 'P12').

    --entrenar (flag, opcional)
        Incluir la etapa de entrenamiento. Por defecto se omite.

    --limpiar (flag, opcional)
        Limpiar resultados previos antes de generar nuevos.

Uso por CLI:
    python -m yolo_mslesseg.ejecutar_pipeline \
        --plano axial \
        --modalidad FLAIR \
        --num_cortes P50 \
        --mejora HE \
        --epochs 50 \
        --completo
"""

import argparse
import logging
import sys
from pathlib import Path

from yolo_mslesseg.scripts.eval import ejecutar_eval_pipeline
from yolo_mslesseg.scripts.extraer_dataset import ejecutar_dataset_pipeline
from yolo_mslesseg.scripts.generar_consenso import ejecutar_consenso_pipeline
from yolo_mslesseg.scripts.generar_predicciones import ejecutar_predicciones_pipeline
from yolo_mslesseg.scripts.promediar_folds import ejecutar_promediar_folds_pipeline
from yolo_mslesseg.scripts.reconstruir_volumen import ejecutar_reconstrucciones_pipeline
from yolo_mslesseg.scripts.setup import ejecutar_setup_pipeline
from yolo_mslesseg.scripts.train import ejecutar_train_pipeline
from yolo_mslesseg.utils.Modelo import Modelo
from yolo_mslesseg.utils.Paciente import Paciente
from yolo_mslesseg.utils.configurar_logging import configurar_logging, get_logger
from yolo_mslesseg.utils.utils import (
    int_o_percentil,
    volumenes_predichos_completos,
    verificar_volumenes_grupo,
    existe_modelo_entrenado,
    calcular_fold,
)

# Configurar logger
configurar_logging(level=logging.INFO, log_file="pipeline.log")
logger = get_logger(__file__)


# =============================
#     FUNCIONES AUXILIARES
# =============================


def verificar_folds_consenso(modelo, epochs, k_folds):
    """
    Verifica qué folds del modelo tienen volúmenes predichos completos
    en los tres planos (axial, coronal y sagital), condición necesaria
    para generar el consenso.
    """
    folds_validos = []
    folds_incompletos = []

    for fold in range(1, k_folds + 1):
        pred_vols_fold_dir = (
            Path("pred_vols") / f"{modelo.base_path}_{epochs}epochs" / f"fold{fold}"
        )
        if verificar_volumenes_grupo(pred_vols_fold_dir):
            folds_validos.append(fold)
        else:
            folds_incompletos.append(fold)

    return folds_validos, folds_incompletos


# ====================================
#   FUNCIONES DE EJECUCIÓN POR ETAPA
# ====================================


def ejecutar_setup(limpiar):
    """ """
    logger.header(
        f"\n📦 Descargando dataset MSLesSeg y preparando estructura de directorios"
    )
    ejecutar_setup_pipeline(limpiar=limpiar)


def ejecutar_dataset(modelo, paciente, k_folds, limpiar):
    """
    Ejecuta la etapa de generación del dataset. Gestiona la extracción
    de cortes y de anotaciones para un paciente individual o para todos
    los pacientes del experimento.
    """
    logger.header(f"\n🧩 Preparando dataset YOLO")
    ejecutar_dataset_pipeline(
        modelo=modelo,
        paciente=paciente,
        k_folds=k_folds,
        limpiar=limpiar,
    )


def ejecutar_train(modelo, epochs, k_folds, entrenar, limpiar):
    """
    Ejecuta la etapa de entrenamiento del modelo YOLO para cada fold.
    Esta etapa es opcional y solo se activa cuando `--entrenar` está presente.
    """
    logger.header(f"\n🧠 Entrenando modelo")

    if not entrenar:
        logger.info("⏹️ Entrenamiento omitido (usar --entrenar para activarlo).")
        return

    for fold_test in range(1, k_folds + 1):
        if existe_modelo_entrenado(modelo, epochs, fold_test):
            logger.skip(f"⏩ Modelo entrenado para fold {fold_test} ya existente.")
        else:
            print(f"\n--- Fold {fold_test} ---\n")
            ejecutar_train_pipeline(
                modelo=modelo,
                fold_test=fold_test,
                epochs=epochs,
                limpiar=limpiar,
            )


def ejecutar_predicciones(modelo, epochs, k_folds, paciente, limpiar):
    """
    Ejecuta la generación de predicciones 2D binarias para un paciente
    individual o para todos los pacientes del experimento.
    """
    logger.header(f"\n🎯 Generando predicciones")

    if paciente is not None:
        ejecutar_predicciones_pipeline(
            modelo=modelo,
            epochs=epochs,
            k_folds=k_folds,
            paciente=paciente,
            limpiar=limpiar,
        )
        return

    else:  # Ejecución completa
        for fold in range(1, k_folds + 1):
            ejecutar_predicciones_pipeline(
                modelo=modelo,
                epochs=epochs,
                k_folds=k_folds,
                fold_test=fold,
                limpiar=limpiar,
            )


def ejecutar_reconstrucciones(modelo, epochs, k_folds, paciente, limpiar):
    """
    Ejecuta la reconstrucción de volúmenes 3D a partir de las predicciones 2D.
    Puede utilizarse para un paciente específico o para todos los pacientes
    del experimento.
    """
    if paciente is not None:
        logger.header(f"\n🧱 Reconstruyendo volumen ({modelo.plano})")
        ejecutar_reconstrucciones_pipeline(
            modelo=modelo,
            epochs=epochs,
            k_folds=k_folds,
            paciente=paciente,
            limpiar=limpiar,
        )

    else:  # Ejecución completa
        logger.header(f"\n🧱 Reconstruyendo volúmenes ({modelo.plano})")
        for fold in range(1, k_folds + 1):
            ejecutar_reconstrucciones_pipeline(
                modelo=modelo,
                epochs=epochs,
                k_folds=k_folds,
                fold_test=fold,
                limpiar=limpiar,
            )


def ejecutar_eval(modelo, epochs, k_folds, paciente, limpiar):
    """
    Ejecuta el cálculo de métricss de evaluación (DSC, AUC, precision,
    recall) sobre los volúmenes reconstruidos, para un paciente
    individual o para todos los pacientes del experimento.
    """
    logger.header(f"\n📈 Calculando métricas ({modelo.plano})")

    if paciente is not None:
        ejecutar_eval_pipeline(
            modelo=modelo,
            epochs=epochs,
            k_folds=k_folds,
            paciente=paciente,
            limpiar=limpiar,
        )
        return

    else:  # Ejecución completa
        for fold in range(1, k_folds + 1):
            ejecutar_eval_pipeline(
                modelo=modelo,
                epochs=epochs,
                k_folds=k_folds,
                fold_test=fold,
                limpiar=limpiar,
            )


def ejecutar_consenso(modelo, epochs, k_folds, paciente, umbral_consenso, limpiar):
    """
    Ejecuta la generación del volumen de consenso y el cálculo de sus métricas,
    para un paciente individual o para todos los pacientes del experimento.
    """
    if paciente is not None:
        fold_paciente = calcular_fold(
            paciente_id=paciente.id,
            k_folds=k_folds,
        )

        pred_vols_root = (
            Path("pred_vols")
            / f"{modelo.base_path}_{epochs}epochs"
            / f"fold{fold_paciente}"
            / paciente.id
        )

        if not volumenes_predichos_completos(pred_vols_root):
            logger.warning(f"\n⚠️ Omitiendo consenso: faltan volúmenes predichos.")
            return False

        logger.header("\n🤝 Generando consenso")
        ejecutar_consenso_pipeline(
            modelo=modelo,
            epochs=epochs,
            k_folds=k_folds,
            umbral=umbral_consenso,
            paciente=paciente,
            limpiar=limpiar,
        )

        logger.header("\n📈 Calculando métricas (consenso)")
        ejecutar_eval_pipeline(
            modelo=modelo,
            plano="consenso",
            epochs=epochs,
            k_folds=k_folds,
            paciente=paciente,
            limpiar=limpiar,
        )

        return True

    else:  # Ejecución completa
        folds_validos, folds_incompletos = verificar_folds_consenso(
            modelo=modelo,
            epochs=epochs,
            k_folds=k_folds,
        )
        consenso_completo = len(folds_validos) == k_folds

        if not consenso_completo:
            logger.warning(
                f"\n⚠️ Omitiendo consenso: faltan volúmenes predichos en fold(s) "
                f"{', '.join(map(str, folds_incompletos))}."
            )
            return False

        logger.header("\n🤝 Generando consenso")
        for fold in folds_validos:
            ejecutar_consenso_pipeline(
                modelo=modelo,
                epochs=epochs,
                k_folds=k_folds,
                umbral=umbral_consenso,
                fold_test=fold,
                limpiar=limpiar,
            )

        logger.header("\n📈 Calculando métricas (consenso)")
        for fold in folds_validos:
            ejecutar_eval_pipeline(
                modelo=modelo,
                plano="consenso",
                epochs=epochs,
                k_folds=k_folds,
                fold_test=fold,
                limpiar=limpiar,
            )

        return True


def ejecutar_promediar_folds(modelo, epochs, k_folds, consenso_generado, limpiar):
    """
    Calcula los resultados globales del experimento promediando los resultados
    de cada fold. Si el consenso fue generado, también promedia sus métricas.
    """
    logger.header(f"\n🧮 Promediando folds ({modelo.plano})")
    ejecutar_promediar_folds_pipeline(
        modelo=modelo,
        epochs=epochs,
        k_folds=k_folds,
        limpiar=limpiar,
    )

    if consenso_generado:
        logger.header(f"\n🧮 Promediando folds (consenso)")
        ejecutar_promediar_folds_pipeline(
            modelo=modelo,
            plano="consenso",
            epochs=epochs,
            k_folds=k_folds,
            limpiar=limpiar,
        )


# =============================
#        FLUJO PRINCIPAL
# =============================


def ejecutar_pipeline(
    modelo,
    epochs,
    umbral_consenso,
    k_folds=5,
    paciente=None,
    completo=None,
    entrenar=False,
    limpiar=False,
):
    """
    Ejecuta el flujo de trabajo completo. Incluye setup, extracción del
    dataset, entrenamiento, predicción, reconstrucción, evaluación,
    consenso y promedio de folds.

    - Modo paciente → ejecuta el flujo para un único paciente en todas las etapas.
    - Modo completo → ejecuta el flujo para todos los folds del dataset en todas las etapas.
    """
    if paciente is not None:
        logger.header(
            f"\n🚀 Iniciando pipeline individual para {paciente.id} "
            f"(modelo = {modelo.model_string}, epochs = {epochs})"
        )
    else:
        logger.header(
            f"\n🚀 Iniciando pipeline completo "
            f"(modelo = {modelo.model_string}, epochs = {epochs})"
        )

    if limpiar:
        logger.info("\n♻️ Limpiando ejecución previa.")

    # --- ETAPA 0 ---
    ejecutar_setup(limpiar)

    # --- ETAPA 1 ---
    ejecutar_dataset(modelo, paciente, k_folds, limpiar)

    # --- ETAPA 2 ---
    ejecutar_train(modelo, epochs, k_folds, entrenar, limpiar)

    # --- ETAPA 3 ---
    ejecutar_predicciones(modelo, epochs, k_folds, paciente, limpiar)

    # --- ETAPA 4 ---
    ejecutar_reconstrucciones(modelo, epochs, k_folds, paciente, limpiar)

    # --- ETAPA 5 ---
    ejecutar_eval(modelo, epochs, k_folds, paciente, limpiar)

    # --- ETAPA 6 ---
    consenso_generado = ejecutar_consenso(
        modelo, epochs, k_folds, paciente, umbral_consenso, limpiar
    )

    # --- ETAPA 7 ---
    if completo:
        ejecutar_promediar_folds(modelo, epochs, k_folds, consenso_generado, limpiar)

    logger.header("\n🏁 Pipeline finalizado correctamente")


# =============================
#       CLI Y EJECUCIÓN
# =============================


def parsear_args(argv=None):
    """
    Parsea los argumentos del script.
    Si no se pasa una lista de argumentos, se leen los utilizados
    al ejecutar el script desde la terminal.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Ejecutar el flujo de trabajo YOLO-MSLesSeg completo.",
    )
    parser.add_argument(
        "--plano",
        type=str,
        required=True,
        choices=["axial", "coronal", "sagital"],
        metavar="[axial, coronal, sagital]",
        help="Plano anatómico de extracción.",
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
        help="Número de cortes a extraer (valor entero o percentil).",
    )
    parser.add_argument(
        "--mejora",
        type=str,
        default=None,
        choices=["HE", "CLAHE", "GC", "LT"],
        metavar="[HE, CLAHE, GC, LT]",
        help="Algoritmo de mejora de imagen a aplicar. Por defecto None.",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        metavar="<k_folds>",
        help="Número de folds para validación cruzada. Por defecto 5.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        metavar="<epochs>",
        help="Número de épocas de entrenamiento.",
    )
    parser.add_argument(
        "--umbral_consenso",
        type=int,
        default=2,
        choices=[2, 3],
        metavar="<umbral_consenso>",
        help="Umbral elegido para generar el consenso por votación mayoritaria (2 o 3). Por defecto 2.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--completo",
        action="store_true",
        help="Ejecutar el flujo de trabajo completo sobre todos los pacientes del dataset.",
    )
    group.add_argument(
        "--paciente_id",
        type=str,
        metavar="<paciente_id>",
        help="Extraer el dataset YOLO solo para el paciente indicado.",
    )
    parser.add_argument(
        "--entrenar",
        action="store_true",
        help="Incluir la etapa de entrenamiento. Por defecto se omite.",
    )
    parser.add_argument(
        "--limpiar",
        action="store_true",
        default=False,
        help="Limpiar todos los resultados generados previamente.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    """
    Función principal del script `ejecutar_pipeline.py`.

    Parsea los argumentos proporcionados por línea de comandos, construye las
    instancias de `Modelo` y (opcionalmente) `Paciente`, y delega la ejecución
    completa del flujo en `ejecutar_pipeline`.
    """
    args = parsear_args(argv)

    modelo = Modelo(
        plano=args.plano,
        num_cortes=args.num_cortes,
        modalidad=args.modalidad,
        k_folds=args.k_folds,
        mejora=args.mejora,
    )

    paciente = (
        Paciente(
            id=args.paciente_id,
            plano=modelo.plano,
            modalidad=modelo.modalidad,
            mejora=modelo.mejora,
        )
        if args.paciente_id is not None
        else None
    )
    try:
        ejecutar_pipeline(
            modelo=modelo,
            epochs=args.epochs,
            umbral_consenso=args.umbral_consenso,
            k_folds=args.k_folds,
            paciente=paciente,
            completo=args.completo,
            entrenar=args.entrenar,
            limpiar=args.limpiar,
        )
    except Exception as e:
        logger.error(f"❌ Error crítico en el pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
