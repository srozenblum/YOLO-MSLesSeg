# 🧪 Demo del proyecto

Esta carpeta contiene una demostración simplificada del _pipeline_.
El objetivo es permitir una ejecución rápida, controlada y completamente autónoma sin
necesidad de entrenar modelos, descargar el _dataset_ completo ni configurar experimentos avanzados.

La demo permite:




> ℹ️ La demo es completamente autónoma, pero conserva la misma lógica de funcionamiento del pipeline completo.
> Para más detalles sobre el sistema general, consultar el [README](../README.md) de la raíz del repositorio.

## 📁 Contenido de la carpeta

La carpeta `demo/` contiene los elementos para una demostración
autónoma incluyendo el dataset de entrada
y los modelos preentrenados.

```
demo/
├── ejecutar_demo.py          # Script principal de ejecución de la demo
│
├── 📁 MSLesSeg-Dataset/                        # Dataset de entrada crudo, descargado desde el repositorio oficial
│
├── 📁 datasets/                                # Datasets YOLO (*️⃣)
│
├── 📁 trains/                                  # Modelos entrenados
│
├── 📁 pred_vols/                               # Volúmenes predichos 3D (*️⃣)
│
├── 📁 results/                                 # Métricas de evaluación (*️⃣)
│
├── 📁 GT/                                      # Volúmenes ground truth (*️⃣)
│
├── 📁 visualizaciones/                         # GIFs y figuras de predicciones 2D (*️⃣)
│
└── README_demo.md                
```

> ℹ️ Las carpetas marcadas con *️⃣ se generan automáticamente durante la ejecución.

## ▶️ Instrucciones de ejecución

Desde la carpeta raíz del repositorio, ejecutar la demo con el siguiente comando:

EN ESTE CASO NO HAY PARAMETROS CLI PORQUE SE FIJAN POR DEFECTO

## Resultados de ejecución

Tras la ejecución, se puede comprobar que se obtienen los siguientes GIFs
dentro de la carpeta `visualizaciones/`:

<p align="center">
  <img src="visualizaciones/Control/FLAIR_P50c_5folds_50epochs/fold2/P14/sagital/P14_FLAIR.gif" height="270">
<img src="visualizaciones/HE/FLAIR_P50c_5folds_50epochs/fold2/P18/axial/P18_FLAIR.gif" height="270">
</p>
