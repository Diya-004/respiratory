# Final Major Project Report Draft

## Declaration

To be filled with student name, roll number, department, college name, guide name, and official declaration text required by your institution.

## Acknowledgments

To be personalized with thanks to your guide, department, institution, and anyone who supported the project work.

## Abstract

Respiratory diseases such as Asthma, Chronic Obstructive Pulmonary Disease (COPD), and Pneumonia are major causes of morbidity and require early screening for timely medical intervention. In many practical situations, especially in resource-constrained environments, access to specialist respiratory assessment is limited. This project presents `PneumoScan`, an AI-assisted respiratory disease screening platform that analyzes recorded lung and breathing sounds to support early respiratory condition screening. The project was developed as an end-to-end system rather than only a standalone classification model. It includes a machine learning pipeline for respiratory audio preprocessing and model training, a web platform for browser-based analysis, and a native Android mobile application for portable use.

The system preprocesses respiratory audio by trimming silence, normalizing the signal, segmenting recordings into fixed-duration windows, and converting them into log-mel spectrogram representations. Multiple deep learning model families were explored within a reusable training framework, including baseline CNN, custom CNN, strong CNN, EfficientNetB0, and CNN-LSTM. The final saved deployment-oriented configuration uses a strong CNN with focal loss, class reweighting, recording-level window aggregation, and Grad-CAM style explainability. The dataset preparation pipeline also includes duplicate removal, label normalization, patient-aware data splitting, and pneumonia-focused augmentation to address class imbalance and improve clinically important minority-class performance.

The final platform supports audio upload or live recording, predicts one of four classes namely Asthma, COPD, Normal, and Pneumonia, estimates severity, produces window-level prediction insights, and displays a heatmap for explainability. On the saved recording-level test evaluation, the final model achieved an accuracy of approximately `81.40%` and a weighted F1-score of approximately `80.75%`, with the best performance observed for COPD. The project demonstrates the feasibility of building a practical AI-assisted respiratory screening platform that combines machine learning, web deployment, and mobile accessibility. Although the system is not a substitute for clinical diagnosis, it provides a strong foundation for future medical decision-support research and deployment.

## Table of Contents

To be generated after final formatting is fixed.

## List of Tables

To be generated after final formatting is fixed.

## List of Figures

To be generated after final formatting is fixed.

## Abbreviations

| Abbreviation | Expanded Form |
| --- | --- |
| AI | Artificial Intelligence |
| COPD | Chronic Obstructive Pulmonary Disease |
| CNN | Convolutional Neural Network |
| CNN-LSTM | Convolutional Neural Network with Long Short-Term Memory |
| MFCC | Mel Frequency Cepstral Coefficient |
| FFT | Fast Fourier Transform |
| API | Application Programming Interface |
| WAV | Waveform Audio File Format |
| UI | User Interface |
| Grad-CAM | Gradient-weighted Class Activation Mapping |

## Chapter 1: Introduction

### 1.1 Introduction to the Topic

Respiratory diseases are among the most common health problems worldwide and frequently require timely screening to reduce complications. Conventional diagnosis often depends on clinical examination, imaging, laboratory testing, and interpretation of respiratory sounds by skilled professionals. However, in low-resource settings or during early screening stages, specialist interpretation may not always be immediately available. Advances in machine learning and audio signal processing have made it possible to analyze lung and breathing sounds computationally and assist in the identification of abnormal respiratory patterns.

This project addresses that opportunity by building an AI-assisted respiratory screening platform that analyzes respiratory audio recordings and predicts probable respiratory conditions. The work combines signal processing, deep learning, explainable AI, web development, and Android application development into a single system. Instead of limiting the scope to offline model training, the project delivers a practical platform where users can upload or record breathing audio and receive interpretable screening output.

### 1.2 Problem Definition

Manual respiratory sound assessment is useful but subjective and depends on the availability and expertise of clinicians. Public respiratory audio datasets are often heterogeneous, imbalanced, and collected under different conditions, which makes model development challenging. Many academic solutions also stop at model training and do not provide complete deployment-ready applications.

The problem addressed in this project is therefore:

To design and develop an AI-assisted platform capable of processing respiratory sound recordings, predicting common respiratory conditions, and presenting the result through user-friendly web and mobile interfaces with interpretable outputs.

### 1.3 Objectives and Outcomes

The main objectives of the project are:

1. To collect and prepare respiratory sound data from multiple sources.
2. To preprocess respiratory audio into a suitable representation for machine learning.
3. To train and evaluate deep learning models for multiclass respiratory disease screening.
4. To reduce data leakage by using patient-aware train, validation, and test splitting.
5. To improve minority-class handling, especially for pneumonia-focused screening.
6. To integrate the trained model into a web interface and an Android mobile application.
7. To provide interpretable outputs such as severity estimation, window-level predictions, and heatmap-based explainability.

Project outcomes achieved:

- a working dataset preparation and training pipeline
- a browser-based respiratory screening interface
- a native Android client for respiratory sound capture and analysis
- saved model metrics and deployment-oriented inference logic
- Grad-CAM style explainability support

### 1.4 Motivation

The motivation for this project comes from the importance of early respiratory disease screening and the growing availability of affordable digital recording devices and smartphones. A portable AI-assisted system can help extend preliminary respiratory screening support beyond specialized hospital environments. The project is also motivated by the need to transform a pure machine learning prototype into a complete engineering solution with accessible interfaces and meaningful outputs.

### 1.5 Thesis Organization

This report is organized as follows:

- Chapter 1 introduces the project background, problem statement, objectives, motivation, and report organization.
- Chapter 2 presents a literature review of related work in respiratory sound classification and AI-assisted screening.
- Chapter 3 describes the proposed system, methodology, architecture, implementation, and results.
- Chapter 4 presents the conclusion and future scope of the project.

## Chapter 2: Literature Review

Note: this chapter should be finalized using five properly cited research papers. I can complete this after you share whether you want IEEE style, APA style, or your college's prescribed citation format.

### 2.1 Paper 1

Placeholder for paper summary:

- problem addressed
- dataset used
- methodology
- results
- limitations

### 2.2 Paper 2

Placeholder for paper summary:

- problem addressed
- dataset used
- methodology
- results
- limitations

### 2.3 Paper 3

Placeholder for paper summary:

- problem addressed
- dataset used
- methodology
- results
- limitations

### 2.4 Paper 4

Placeholder for paper summary:

- problem addressed
- dataset used
- methodology
- results
- limitations

### 2.5 Paper 5

Placeholder for paper summary:

- problem addressed
- dataset used
- methodology
- results
- limitations

### 2.6 Summary Table on Literature Survey

To be filled after selecting the five final papers.

### 2.7 Gaps Identified

The following research and implementation gaps motivated this project:

- many prior works emphasize model accuracy but do not provide complete end-to-end usable platforms
- data leakage is not always carefully handled in respiratory audio studies
- explainability is often limited or absent
- recording-level prediction is sometimes ignored in favor of isolated clip-level classification
- minority classes such as pneumonia often remain underrepresented
- mobile deployment and practical usability are frequently not addressed

## Chapter 3: Proposed System

### 3.1 Introduction

The proposed system is an AI-assisted respiratory disease screening platform named `PneumoScan`. It was developed as a layered solution consisting of data preparation, preprocessing, deep learning model training, deployment-oriented inference, a browser-based interface, and a native Android application. The project initially started with a FastAPI-based MVP to validate the idea, and later evolved into a more structured rebuild with a real inference backend and a mobile companion app.

### 3.2 Methodology

The project methodology consists of the following stages:

1. Collection and preparation of respiratory audio data from multiple sources.
2. Deduplication and metadata preparation using hashing and normalized labels.
3. Patient-aware split generation for train, validation, and test sets.
4. Audio trimming, normalization, and fixed-duration segmentation.
5. Log-mel spectrogram generation with optional delta and delta-delta channels.
6. Model training using different architectures under a common training framework.
7. Evaluation on validation and untouched test sets.
8. Recording-level window aggregation and explainability generation.
9. Integration into browser and Android interfaces.

#### 3.2.1 Dataset Preparation

The paper-oriented dataset audit saved with the project shows:

- Total deduplicated recordings: `1756`
- Classes:
  - Asthma: `96`
  - COPD: `1330`
  - Normal: `163`
  - Pneumonia: `167`
- Sources:
  - chest_wall
  - icbhi
  - respdb_tr

Duplicate removal was performed using SHA-256 file hashing. Labels from multiple sources were normalized into four final target classes: Asthma, COPD, Normal, and Pneumonia.

#### 3.2.2 Split Strategy

To reduce train-test leakage, the project uses patient-aware splitting. Samples belonging to the same patient are grouped and assigned together to one split. The split logic is also source-aware within each label group to preserve class distribution and reduce bias.

#### 3.2.3 Audio Preprocessing

Each respiratory audio sample is:

- loaded and resampled
- silence-trimmed
- amplitude-normalized
- fitted to a target duration
- converted into a log-mel spectrogram

For the final saved model family, the main preprocessing parameters are:

- sample rate: `4000 Hz`
- clip duration: `5 seconds`
- mel bands: `128`
- FFT size: `512`
- hop length: `128`
- frequency range: `50 Hz` to `2000 Hz`
- delta channels: enabled

#### 3.2.4 Model Development

The training framework supports multiple architectures:

- baseline CNN
- custom CNN
- strong CNN
- EfficientNetB0
- CNN-LSTM

The final saved deployment-oriented configuration uses a `strong_cnn` architecture with focal loss, clipped class weighting, and pneumonia-sensitive inference aggregation.

#### 3.2.5 Pneumonia-Focused Augmentation

To improve minority-class learning for pneumonia, the project includes an additional pneumonia-focused dataset build stage. In the saved summary:

- original training Pneumonia clips: `500`
- final training Pneumonia clips after augmentation: `900`
- total additional augmented rows in this stage: `878`

This step was included because pneumonia is clinically important and relatively harder for the model compared with COPD.

### 3.3 System Architecture

The overall architecture can be described in six layers:

1. Input Layer
   - respiratory audio captured through file upload or microphone recording

2. Data Processing Layer
   - silence trimming
   - normalization
   - fixed-window extraction
   - spectrogram generation

3. Machine Learning Layer
   - trained deep learning classifier
   - class probability estimation
   - recording-level aggregation

4. Explainability Layer
   - representative window selection
   - Grad-CAM style heatmap generation

5. Backend Layer
   - Flask inference API for final deployment flow
   - FastAPI-based earlier MVP endpoint

6. Presentation Layer
   - browser-based web platform
   - Android mobile application

### 3.4 Implementation

#### 3.4.1 Initial Web MVP

The initial prototype consists of:

- FastAPI backend
- static HTML, CSS, and JavaScript frontend
- audio upload endpoint
- feature extraction using `librosa`
- heuristic inference service
- generated report and explanation regions

This MVP was useful for validating the user workflow before integrating the final trained pipeline.

#### 3.4.2 Final Inference Backend

The final backend uses Flask and exposes a `/predict` endpoint. It:

- validates uploaded audio type
- stores the file temporarily
- preprocesses it with the same pipeline used during training
- extracts multiple windows from each recording
- predicts class probabilities per window
- aggregates predictions at recording level
- returns:
  - filename
  - predicted class
  - confidence
  - class probabilities
  - severity
  - heatmap
  - windows used
  - aggregation mode
  - representative window
  - all window predictions

#### 3.4.3 Web Platform

The final browser interface presents the project as `PneumoScan`. It includes:

- modern upload interface
- drag-and-drop style respiratory audio workflow
- visual disease summary
- class-wise probability cards
- severity indication
- spectrogram or heatmap-based visual explanation
- recommendation section

#### 3.4.4 Android Mobile Application

The Android application was developed using Kotlin and Jetpack Compose. It supports:

- live microphone capture
- recorded WAV file generation
- file picker upload
- HTTP communication with the Python backend
- mobile presentation of:
  - prediction
  - confidence
  - severity
  - class probabilities
  - window timeline
  - heatmap image

This mobile module increases the practical usability of the project and distinguishes it from purely desktop or notebook-based research work.

### 3.5 Results and Discussion

#### 3.5.1 Validation Results of the Saved Strong-CNN Pneumonia-Focus Run

| Metric | Value |
| --- | --- |
| Best Validation Accuracy | 0.7389 |
| Best Validation Loss | 0.4388 |
| Macro F1-score | 0.5132 |
| Weighted F1-score | 0.7273 |

Per-class validation F1-score:

| Class | F1-score |
| --- | --- |
| Asthma | 0.3846 |
| COPD | 0.8714 |
| Normal | 0.4333 |
| Pneumonia | 0.3636 |

#### 3.5.2 Recording-Level Test Results

| Metric | Value |
| --- | --- |
| Accuracy | 0.8140 |
| Macro Precision | 0.5817 |
| Macro Recall | 0.5862 |
| Macro F1-score | 0.5791 |
| Weighted F1-score | 0.8075 |

Per-class recording-level test F1-score:

| Class | F1-score |
| --- | --- |
| Asthma | 0.5625 |
| COPD | 0.9243 |
| Normal | 0.6190 |
| Pneumonia | 0.2105 |

#### 3.5.3 Discussion

The results show that the final system performs strongly on COPD and demonstrates good overall weighted performance at recording level. This indicates that the system can capture dominant respiratory patterns well when recording-window aggregation is used. The performance on Asthma and Normal classes is moderate, while Pneumonia remains the most challenging class. This suggests that additional pneumonia data, improved class balancing, or model ensembles may be required to raise sensitivity for that class.

One important outcome of this project is that it demonstrates a complete deployable workflow rather than only offline classification. The browser-based platform and Android application both consume model output and present the result in a more clinically understandable format. The use of heatmaps and window-level predictions also improves transparency compared with black-box classification alone.

## Chapter 4: Conclusion and Future Scope

### 4.1 Conclusion

This project successfully designed and developed an AI-assisted respiratory disease screening platform based on lung and breathing sound analysis. The work integrates data preparation, audio preprocessing, deep learning model training, explainable inference, a browser-based web interface, and a native Android mobile application into a single practical system. The final saved model configuration achieved promising recording-level test performance, especially for COPD, while the platform also delivered useful supporting outputs such as severity estimation and visual explainability.

The project demonstrates that respiratory sound analysis can be transformed from a research pipeline into a user-facing screening platform. Although the system is not intended to replace clinical diagnosis, it provides a strong technical foundation for future decision-support tools in respiratory health screening.

### 4.2 Future Scope

Future enhancements can include:

1. Collecting larger and more balanced respiratory sound datasets, especially for pneumonia and asthma.
2. Improving model robustness against noise and variable recording quality.
3. Exploring ensemble models or transformer-based audio architectures.
4. Integrating additional patient metadata such as age, symptoms, temperature, and oxygen saturation.
5. Adding secure cloud deployment and authenticated user management.
6. Supporting digital stethoscope devices and IoT-based edge capture workflows.
7. Generating downloadable PDF reports and patient history records.
8. Conducting clinical validation with domain experts and real hospital data.

## References

To be added after you confirm the citation format and the five literature-review papers to use.

## List of Publications

Add only if your college requires published papers, conference abstracts, or submitted manuscripts related to the project.

