> Trabajo de Fin de Grado  
> Autor: Sebastián Rozenblum  
> Tutores: Miguel Ángel Molina Cabello, Paula Ariadna Jiménez Partinen  
> Ingeniería de la Salud · Mención en Bioinformática  
> Universidad de Málaga · Curso 2025–2026

# 🧠💻 YOLO-MSLesSeg: segmentación automática de lesiones de esclerosis múltiple con YOLOv11-seg

## 📄 Descripción general

Este proyecto implementa un pipeline completo de segmentación y evaluación de lesiones de esclerosis múltiple en
imágenes de resonancia magnética utilizando modelos YOLOv11-seg. El objetivo es proporcionar
una herramienta reproducible que permita identificar y cuantificar lesiones de forma consistente,
reduciendo la variabilidad asociada a la segmentación manual.

El flujo está diseñado para ejecutarse de forma modular y escalable, permitiendo:

- Procesar volúmenes médicos en distintos planos anatómicos y modalidades de resonancia magnética (T1, T2, FLAIR).
- Aplicar técnicas de mejora de imagen para optimizar el contraste y la detección de lesiones.
- Generar segmentaciones automáticas a nivel de corte con modelos YOLOv11-seg entrenados específicamente para cada
  configuración experimental.
- Gestionar configuraciones flexibles por paciente o para el conjunto completo.
- Integrar salidas intermedias en una estructura organizada y reproducible que facilita análisis posteriores.
- Evaluar cuantitativamente el rendimiento del modelo mediante métricas estandarizadas en el ámbito de la segmentación
  médica.

---

## ⛓️ Flujo general del *pipeline*

El proceso completo consta de ocho etapas secuenciales,
automatizadas mediante el script `ejecutar_pipeline.py`:

0. **Descarga del dataset oficial YOLOMSLesSeg y preparación de la estructura de directorios**: `setup.py`
1. **Extracción del dataset YOLO con imágenes y anotaciones**: `extraer_dataset.py`
2. **Entrenamiento del modelo YOLOv11-seg**: `train.py`
3. **Generación de predicciones bidimensionales individuales**: `generar_predicciones.py`
5. **Reconstrucción de volúmenes predichos**: `reconstruir_volumen.py`
4. **Combinar volúmenes predichos en distintos planos (consenso)**: `generar_consenso.py`
6. **Evaluación y métricas de rendimiento**: `eval.py`
7. **Cálculo de resultados globales**: `promediar_folds.py`

Cada módulo puede ejecutarse de forma independiente o a través del *pipeline* global,
lo que garantiza flexibilidad para depuración o experimentación.

---

## 🗂️ Estructura del repositorio

El repositorio se organiza de la siguiente manera:

```
📁 YOLO-MSLesSeg/                  
│
├── 📁 yolo_mslesseg/                           # Paquete principal del proyecto
│   │ 
│   ├── ejecutar_pipeline.py                    # Script para ejecutar el pipeline completo
│   │
│   ├── 📁 configs/                             # Clases de configuración por etapa
│   │   ├── ConfigSetUp.py
│   │   ├── ConfigTrain.py
│   │   ├── ConfigPred.py
│   │   ├── ConfigRecVol.py
│   │   ├── ConfigEval.py
│   │   └── ConfigConsenso.py
│   │
│   ├── 📁 scripts/                             # Scripts ejecutables que componen el pipeline
│   │   ├── setup.py
│   │   ├── extraer_dataset.py
│   │   ├── train.py
│   │   ├── generar_predicciones.py
│   │   ├── reconstruir_volumen.py
│   │   ├── generar_consenso.py
│   │   ├── eval.py
│   │   └── promediar_folds.py
│   │
│   ├── 📁 utils/                               # Scripts auxiliares y clases base
│   │
│   └── 📁 extras/                              # Scripts adicionales pero no esenciales
│
├── 📁 demo/                                    # Ejecuciones reducidas del pipeline para demostración simple
│
├── 📁 MSLesSeg-Dataset/                        # Dataset de entrada crudo, descargado desde el repositorio oficial
│
├── 📁 datasets/                                # Datasets YOLO
│
├── 📁 trains/                                  # Modelos entrenados
│
├── 📁 vols/                                    # Volúmenes predichos 3D
│
├── 📁 results/                                 # Métricas de evaluación
│
├── 📁 GT/                                      # Volúmenes ground truth
│
├── requirements.txt
└── README.md
```

---

## 🖥️ Requisitos del sistema

Para ejecutar correctamente el proyecto se requiere el siguiente entorno básico:

### Python

- Python **3.10** o superior.

### Hardware

- **GPU NVIDIA** con soporte **CUDA** (opcional pero recomendada para entrenamiento).
- **CPU de múltiples núcleos** si no se dispone de GPU.
- **8–16 GB RAM** mínimos para manejar volúmenes NIfTI.
- **3–6 GB** de espacio libre para datasets, modelos y predicciones.

### Software y frameworks

- PyTorch (con soporte CUDA si se usa GPU).
- Ultralytics YOLOv11-seg.
- OpenCV, NumPy, NiBabel, Matplotlib y demás dependencias listadas en `requirements.txt`.

### Sistemas operativos compatibles

- macOS (Apple Silicon)
- Linux (Ubuntu recomendado)
- Windows (compatible mediante WSL2)

---

## ⚙️ Configuración del entorno

### 1. Ubicarse en la carpeta raíz del proyecto

Antes de ejecutar cualquier comando, situarse en la carpeta raíz del proyecto:

```bash
cd YOLO-MSLesSeg
```

### 2. Crear y activar entorno virtual

Se recomienda crear un **entorno virtual** dedicado al proyecto para evitar
conflictos con otras instalaciones de Python y asegurar una ejecución limpia y reproducible.

#### macOS/Linux

```bash
python3 -m venv venv_mslesseg
source venv/bin/activate
```

#### Windows (PowerShell)

```bash
python3 -m venv venv_mslesseg
venv\Scripts\activate
```

### 3. Instalar dependencias

```
pip install -r requirements.txt
```

### 4.Instalar PyTorch con GPU (_opcional_)

Si el sistema utilizado tiene una GPU NVIDIA compatible, es posible instalar PyTorch con CUDA siguiendo
las [instrucciones oficiales](https://pytorch.org/get-started/locally/).
Por defecto, la instalación funcionará en **CPU**, suficiente para predicción y evaluación.

---

## 🚀 Ejecución del _pipeline_

Una vez configurado el entorno, el _pipeline_ completo puede ejecutarse con un único comando desde la carpeta raíz del
proyecto:

```bash
python -m yolo_mslesseg.ejecutar_pipeline \
    --plano "axial" \
    --modalidad "FLAIR" \
    --mejora "CLAHE" \
    --num_cortes P50 \
    --epochs 50 \
    -- completo
```

Este comando ejecuta automáticamente todas las fases del flujo.
Los resultados se almacenan en la carpeta `results/`, siguiendo la estructura definida por el experimento.

### Parámetros de ejecución

Los siguienes argumentos permiten personalizar la ejecución de `ejecutar_pipeline.py`
y llevar a cabo experimentos para distintas combinaciones de parámetros:

| Argumento           | Tipo / Valores                             | Descripción                                                         |
|---------------------|--------------------------------------------|---------------------------------------------------------------------|
| `--plano`           | `axial`, `coronal`, `sagital`              | Plano anatómico del modelo.                                         |
| `--modalidad`       | `T1`, `T2`, `FLAIR` (múltiples permitidas) | Modalidad(es) de extracción. Por defecto, todas.                    |
| `--num_cortes`      | Entero o percentil (`PXX`)                 | Número de cortes a extraer. Acepta valores como `20`, `P50`, `P75`. |
| `--mejora`          | `HE`, `CLAHE`, `GC`, `LT`                  | Algoritmo de mejora de imagen. Por defecto, ninguno.                |
| `--k_folds`         | Entero                                     | Número de folds para validación cruzada. Por defecto, `5`.          |
| `--epochs`          | Entero                                     | Número de épocas de entrenamiento.                                  |
| `--umbral_consenso` | `2` o `3`                                  | Umbral para la votación mayoritaria del consenso. Por defecto, `2`. |
| `--completo`        | Flag                                       | Ejecutar el flujo completo sobre todos los pacientes del dataset.   |
| `--paciente_id`     | ID de paciente (ej. `P12`)                 | Ejecutar el flujo solo para un paciente específico.                 |
| `--entrenar`        | Flag                                       | Incluir la etapa de entrenamiento (omitida por defecto).            |
| `--limpiar`         | Flag                                       | Limpiar todos los resultados generados previamente.                 |

###

###

TABLA DE PARAMETROS CLI (VER SI VA ACA O EN UNA SECCION APARTEº)

---

## Ejecución modular

## 🖼️ Ejemplo visual

EJECUTAR DEMO + PONER ANIMACIÓN

---

## 🔬 Experimental Design & Methodology

### Dataset

### Validación cruzada

### Métricas de rendimiento

---

## 📚 Referencias

	•	Ultralytics YOLOv8 documentation: https://docs.ultralytics.com￼
	•	NIfTI format specification: https://nifti.nimh.nih.gov￼
	•	MRI lesion segmentation benchmarks: LesionSeg 2023, MSSEG-2
