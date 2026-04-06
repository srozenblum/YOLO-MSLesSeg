# 🔬 Methodology and Experimental Design

## 1. Introduction to the Experimental Design

This work adopts an experimental design aimed at systematically evaluating the performance of automatic
multiple sclerosis lesion segmentation models on MRI images.
Following the principles of the scientific method, the experimental design is based on formulating controlled
experiments, systematically varying the factors of interest, and objectively evaluating the results using
quantitative metrics and complementary qualitative analysis. The design has been conceived to be modular,
reproducible, and extensible, enabling results that are robust, quantifiable, and replicable.

---

## 2. MSLesSeg Dataset

The experiments were conducted using the dataset from the **MSLesSeg Competition** (ICPR 2024).
The main characteristics of the dataset are as follows:

- **Patients:** 75 patients diagnosed with multiple sclerosis
  (53 belonging to the training set and 22 to the test set).

- **Timepoints:** variable number per patient (between 1 and 4 acquisitions).

- **MRI modalities:** T1, T2, and FLAIR (Fluid Attenuated Inversion Recovery).

- **Resolution:** isotropic voxels of 1 mm³.

- **Ground truth:** manual segmentations produced by clinical experts.

---

## 3. Preprocessing and Enhancement Algorithms

Since one of the contributions of this work is to analyse the effect of preprocessing on model performance,
several classical image enhancement techniques applied prior to segmentation were evaluated.

The techniques considered were:

- **HE (Histogram Equalization):** redistributes intensities to exploit the full dynamic range of the image,
  increasing global contrast and highlighting details that might otherwise go unnoticed.

- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** divides the image into smaller regions and
  applies limited equalisation to each one, improving contrast in specific areas without excessively
  amplifying noise.

- **GC (Gamma Correction):** applies a non-linear brightness adjustment via a power-law transformation
  I' = I^γ with γ = 2, fixed and non-configurable. This suppresses dark background regions while
  preserving the relative contrast of hyperintense lesions.

- **LT (Linear Transformation):** improves contrast primarily in dark regions by applying a logarithmic
  function to intensity values, compressing the range of the brightest pixels and expanding that of the
  darkest ones.

Each technique was applied independently, generating distinct experimental configurations that allow its
influence to be analysed.

---

## 4. Experiment Configuration

The system has been designed with a fully parametrisable architecture,
allowing almost every aspect of the pipeline to be configured:

- Anatomical processing plane (axial, coronal, or sagittal)
- MRI modalities (T1, T2, FLAIR), as well as any combination thereof
- Number of slices extracted per volume
- Image enhancement technique applied
- Cross-validation scheme with any number of folds, or fixed train/test split
- Number of training epochs

This flexibility enables experiments with different configurations to be conducted
in a systematic and reproducible manner.

### Configuration Used in This Work

To ensure experimental consistency and facilitate direct comparison between enhancement techniques,
**all experiments use the same base configuration**, varying only the preprocessing algorithm applied:

- **Planes:** axial, coronal, and sagittal (all three)
- **Modality:** FLAIR only
- **Timepoint:** first timepoint per patient (T1)
- **Slices extracted:** 50th percentile of the total volume slices
- **Training epochs:** 50
- **Cross-validation:** 5 folds
- **Consensus threshold:** 2 (majority voting across planes)

This fixed configuration makes it possible to isolate the effect of each enhancement technique
(HE, CLAHE, GC, LT, or none) on model performance.

---

## 5. Evaluation Schemes

In order to evaluate the system's performance robustly, two complementary data partitioning schemes
were adopted: patient-level five-fold cross-validation and evaluation under the official train/test split.
The former estimates the model's average performance over the available training set, while the latter
replicates the scenario set out in the original competition, evaluating the system on a completely
independent set.

### Cross-Validation ($k > 1$)

- **Scheme:** $k$-fold cross-validation.
- **Patient-level split:** ensures that volumes from the same patient do not appear simultaneously in the
  training and test sets.
- **Stratified and deterministic assignment:** folds are constructed consecutively and in a balanced manner
  from patient identifiers, without randomisation, to guarantee full reproducibility across runs.
- **Exhaustive evaluation:** each fold acts as the test set exactly once, while the remaining folds are used
  for training, providing $k$ independent evaluations per configuration.

### Fixed Train/Test Split ($k = 1$)

- **Training:** all patients in the training set are used.
- **Evaluation:** performed exclusively on the test set.
- **Full independence:** no data is shared between training and evaluation.
- **Competitive scenario:** replicates the evaluation scheme established in the MSLesSeg competition.

---

## 6. Evaluation Metrics

Segmentation quality was evaluated quantitatively using metrics widely adopted in biomedical segmentation.
Specifically, the following four metrics were used, which assess performance from complementary perspectives:

- **Dice Similarity Coefficient (DSC):** measures the overlap between the predicted mask and the ground truth
  mask; particularly well-suited for segmentation tasks.

- **Area Under the ROC Curve (AUC):** evaluates the model's ability to discriminate between classes at
  different decision thresholds.

- **Precision:** proportion of positive predictions that are correct.

- **Recall:** proportion of true positives correctly identified by the model.

---

## 7. Reproducibility and Model Availability

To guarantee full reproducibility of the experimental results, the trained models can be found in the
`trains/` directory.

Each set of trained weights is clearly associated with a specific experimental configuration, whose
parameters are defined during the model configuration phase and explicitly recorded in the directory
structure and result files generated by the pipeline.

This correspondence ensures complete traceability between experimental configuration, trained model, and
obtained metrics, allowing every experiment to be reproduced exactly without the need to retrain the models.
