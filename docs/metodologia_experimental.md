# 🔬 Metodología y diseño experimental

## 1. Introducción al diseño experimental

Este trabajo adopta un diseño experimental orientado a evaluar de forma sistemática el rendimiento de modelos de
segmentación automática de lesiones de esclerosis múltiple en imágenes de resonancia magnética.
Siguiendo los principios del método científico, el diseño experimental se basa en la formulación de experimentos
controlados, la variación sistemática de los factores de interés y la evaluación objetiva de los resultados mediante
métricas cuantitativas y análisis cualitativos complementarios. El diseño se ha concebido para ser modular,
reproducible y extensible, permitiendo extraer resultados que sean robustos, cuantificables y replicables.

---

## 2. Dataset MSLesSeg

Los experimentos se llevaron a cabo utilizando el conjunto de datos de la **MSLesSeg Competition** (ICPR 2024).
Las principales características del _dataset_ son las siguientes:

- **Pacientes:** 75 pacientes diagnosticados con esclerosis múltiple  
  (53 correspondientes al conjunto de entrenamiento y 22 al conjunto de test). Los experimentos se realizaron
  exclusivamente sobre los pacientes del conjunto de
  entrenamiento, mientras que los pacientes del conjunto de test se mantienen disponibles dentro del _dataset_ de
  entrada, pero no se utilizan para el diseño experimental propuesto.


- **_Timepoints_:** número variable por paciente (entre 1 y 4 adquisiciones).


- **Modalidades de imagen de resonancia magnética:** T1, T2 y FLAIR (Fluid Attenuated Inversion Recovery).


- **Resolución:** vóxeles isotrópicos de 1 mm³.


- **Ground truth:** segmentaciones manuales realizadas por expertos clínicos.

---

## 3. Preprocesamiento y algoritmos de mejora

Dado que una de las contribuciones del trabajo consiste en analizar el efecto del preprocesamiento sobre el desempeño
del modelo, se evaluaron distintas técnicas clásicas de mejora de imagen aplicadas de forma previa a la segmentación.

Las técnicas consideradas fueron:

- **HE (Histogram Equalization):** redistribuye las intensidades para aprovechar todo el rango dinámico de la imagen,
  aumentando el contraste global y resaltando detalles que podrían pasar desapercibidos.


- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** divide la imagen en regiones
  más pequeñas y aplica una ecualización limitada a cada una, lo que permite mejorar el contraste en áreas específicas
  sin amplificar excesivamente el ruido.


- **GC (Gamma Correction):** propone un ajuste no lineal del brillo de la imagen, permitiendo enfatizar regiones oscuras
  o
  brillantes según el valor del parámetro $\gamma$. En la implementación utilizada, el valor de $\gamma$ no es
  parametrizable y se fija en $\gamma = 2$. Esto comprime el rango de intensidades, oscureciendo las regiones medias y
  brillantes


- **LT (Linear Transformation):** mejora el contraste principalmente en las regiones oscuras, aplicando una función
  logarítmica a los valores de intensidad, lo que comprime el rango de los píxeles más brillantes
  y expande el de los más oscuros.

Cada técnica se aplicó de manera independiente, generando configuraciones experimentales diferenciadas que permiten
analizar su influencia.

---

## 4. Configuración de los experimentos

El sistema ha sido diseñado con arquitectura completamente parametrizable,
permitiendo configurar casi todos los aspectos del _pipeline_:

- Plano anatómico de procesamiento (axial, coronal o sagital)
- Modalidades de imagen de resonancia magnética (T1, T2, FLAIR), así como cualquier combinación entre ellas
- Número de cortes extraídos por volumen
- Técnica de mejora de imagen aplicada
- Esquema de validación cruzada con cualquier número de *folds*
- Número de épocas de entrenamiento
- Valor del umbral utilizado para la generación del consenso (2: votación mayoritaria entre planos; 3: unanimidad)

Esta flexibilidad permite realizar experimentos con diferentes configuraciones de forma
sistemática y reproducible.

### Configuración utilizada en este trabajo

Para garantizar la coherencia experimental y facilitar la comparación directa entre
técnicas de mejora de imagen, **todos los experimentos utilizan la misma
configuración base**, variando únicamente el algoritmo de preprocesamiento aplicado:

- **Planos:** axial, coronal y sagital (todos)
- **Modalidad:** FLAIR exclusivamente
- **Timepoint:** primer _timepoint_ de cada paciente (T1)
- **Cortes extraídos:** percentil 50 del total de cortes del volumen
- **Épocas de entrenamiento:** 50
- **Validación cruzada:** 5 *folds*
- **Umbral de consenso:** 2 (votación mayoritaria entre planos)

Esta configuración fija permite
aislar el efecto de cada técnica de mejora de imagen (HE, CLAHE, GC, LT, o ninguna) sobre el
rendimiento del modelo.


---

## 5. Validación cruzada

Para garantizar una evaluación robusta, se empleó un esquema de **validación cruzada a nivel de
paciente**. Las principales características se resumen a continuación:

- **Esquema:** validación cruzada de $k$ _folds_.
- **División a nivel paciente:** asegura que los volúmenes de un mismo paciente no aparezcan simultáneamente en los
  conjuntos de entrenamiento y test.
- **Asignación estratificada y determinista:** la partición en _folds_ se realiza de forma consecutiva y balanceada a
  partir del identificador de paciente, sin aleatorización, con el objetivo de garantizar reproducibilidad completa
  entre ejecuciones.
- **Evaluación exhaustiva:** cada _fold_ actúa como conjunto de test exactamente una vez, mientras que los
  restantes se utilizan para entrenamiento, proporcionando $k$ evaluaciones independientes por configuración.

---

## 6. Métricas de evaluación

La calidad de la segmentación se evaluó cuantitativamente mediante métricas ampliamente utilizadas en el ámbito de la
segmentación biomédica. Concretamente, se utilizaron las siguientes cuatro, que permiten juzgar el rendimiento desde
perspectivas complementarias:

- **Dice Similarity Coefficient (DSC):** mide el solapamiento entre la máscara predicha y la máscara _ground truth_,
  siendo especialmente adecuada para tareas de segmentación.


- **Área bajo la curva ROC (AUC):** evalúa la capacidad del modelo para distinguir entre clases a distintos umbrales de
  decisión.


- **Precision:** proporción de predicciones positivas correctamente realizadas.


- **Recall:** proporción de verdaderos positivos correctamente identificados por el modelo.

---

## 7. Reproducibilidad y disponibilidad de modelos

Con el objetivo de garantizar la reproducibilidad completa de los resultados experimentales, los modelos entrenados
pueden encontrarse en la carpeta `trains/`.

Cada conjunto de pesos entrenados se asocia de forma clara a una configuración experimental concreta, cuyos parámetros
quedan definidos en la fase de configuración del modelo y registrados de manera explícita en la estructura de
directorios y en los archivos de resultados generados por el _pipeline_.

Esta correspondencia garantiza la trazabilidad completa entre configuración experimental, modelo entrenado y métricas
obtenidas, permitiendo reproducir exactamente cada experimento sin necesidad de reentrenar los modelos.
