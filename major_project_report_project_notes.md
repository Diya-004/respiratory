# Major Project Report Notes

## Working Title

PneumoScan: AI-Assisted Respiratory Disease Screening Platform for Early Detection of Respiratory Disorders Using Lung Sound Analysis

## One-Paragraph Project Summary

This major project focuses on building an AI-assisted respiratory screening platform that analyzes recorded lung or breathing sounds and predicts likely respiratory conditions such as Asthma, COPD, Pneumonia, or Normal breathing patterns. The work evolved from an initial web-based MVP into a more complete platform that now includes a structured machine learning pipeline, a browser-based interface, and a native Android mobile application. The final system supports respiratory audio upload or on-device recording, feature preprocessing through log-mel spectrogram generation, deep-learning-based prediction, severity estimation, Grad-CAM style explainability, recording-window analysis, and clinician-friendly output presentation.

## What We Built

### 1. Initial MVP Web Application

Location: `app/` and `static/` at the project root

- FastAPI backend with `/api/analyze` endpoint
- Audio upload support for `.wav`, `.mp3`, `.m4a`, `.flac`, and `.ogg`
- Acoustic feature extraction using `librosa`
- Heuristic baseline model for end-to-end demo flow
- Generated outputs:
  - predicted condition
  - confidence score
  - severity label
  - urgent alert
  - explanation regions
  - diagnostic report
- Static single-page web interface for upload and result visualization

This component is best described in the report as the first working prototype used to validate the end-to-end idea before integrating the final deep-learning inference stack.

### 2. Final ML Rebuild and Real Inference Backend

Location: `respiratory_ai_rebuild/`

This is the main technical core of the final project. It contains:

- dataset preparation scripts
- duplicate removal using file hash
- label normalization across multiple respiratory datasets
- leakage-safe patient-aware train, validation, and test split generation
- spectrogram preprocessing pipeline shared by both training and inference
- multiple model architectures:
  - baseline CNN
  - custom CNN
  - strong CNN
  - EfficientNetB0
  - CNN-LSTM
- training and evaluation scripts
- Grad-CAM style visual explainability
- recording-level windowed prediction aggregation
- Flask backend for deployment-style inference
- polished browser frontend named `PneumoScan`

### 3. Android Mobile Application

Location: `respiratory_ai_android_app/`

- Native Android app built with Kotlin and Jetpack Compose
- Allows two input modes:
  - live microphone recording
  - file selection from device storage
- Runtime permission handling for microphone access
- WAV recording implemented with `AudioRecord`
- Retrofit-based API communication with the Python inference backend
- Mobile dashboard showing:
  - predicted class
  - confidence
  - severity
  - class-wise probabilities
  - recording timeline windows
  - heatmap image

This should be presented in the report as the mobile client layer of the overall platform.

## Final System Flow

1. Respiratory sound is captured from a phone microphone, uploaded audio file, or external recording source.
2. Audio is trimmed, normalized, resampled, and converted into fixed-duration windows.
3. Each window is transformed into a log-mel spectrogram image, optionally with delta and delta-delta channels.
4. The trained deep-learning model predicts class probabilities for each window.
5. Window probabilities are aggregated at recording level.
6. Pneumonia-sensitive post-processing is applied when needed.
7. The backend returns:
   - primary prediction
   - confidence
   - secondary prediction
   - severity
   - Grad-CAM heatmap
   - representative window
   - all window-level predictions
8. The web app and Android app present the results in a user-friendly way.

## Main Technologies Used

### Backend and ML

- Python
- FastAPI for the MVP API
- Flask for the final inference backend
- TensorFlow / Keras
- NumPy
- Pandas
- scikit-learn
- librosa
- soundfile
- matplotlib
- seaborn

### Frontend

- HTML
- CSS
- JavaScript

### Mobile

- Kotlin
- Jetpack Compose
- Retrofit
- OkHttp
- Kotlin Serialization
- Coroutines

## Dataset Notes

The project uses merged respiratory audio data prepared from multiple sources. Based on the saved audit files, the paper-oriented merged dataset contains:

- Total deduplicated recordings: 1756
- Classes:
  - Asthma: 96
  - COPD: 1330
  - Normal: 163
  - Pneumonia: 167
- Sources observed in the prepared paper dataset:
  - chest_wall
  - icbhi
  - respdb_tr

The earlier audit version used for another prepared dataset shows 1252 deduplicated records after removing 4 duplicates. This is useful to mention if you want to describe the dataset refinement process from an earlier merged version to the paper-oriented version.

## Important Data Engineering Contributions

These are strong points to emphasize in the report:

- Duplicate recording removal using SHA-256 hashing
- Label normalization across differently named diagnosis labels
- Patient-aware split generation to reduce train-test leakage
- Source-aware balancing during split creation
- Window-based recording analysis instead of relying on a single short segment
- Shared preprocessing pipeline for both training and inference
- Pneumonia-focused augmentation and source-balancing strategy

## Pneumonia-Focused Dataset Strategy

The final training pipeline includes a pneumonia-focused dataset build stage:

- Base train counts before this stage:
  - Asthma: 507
  - COPD: 4560
  - Normal: 651
  - Pneumonia: 500
- Final train counts after pneumonia-focused augmentation:
  - Asthma: 507
  - COPD: 4560
  - Normal: 651
  - Pneumonia: 900
- Total augmented rows added in this stage: 878

This is a very important discussion point because it shows that the project did not just train a model directly, but also attempted to improve minority-class representation for clinically important pneumonia detection.

## Final Model Configuration Used in the Best Saved Run

From `models_strong_cnn_pneumonia_focus` and the active config:

- Architecture: `strong_cnn`
- Sample rate: `4000 Hz`
- Clip duration: `5 seconds`
- Mel bands: `128`
- FFT size: `512`
- Hop length: `128`
- Delta channels enabled: `true`
- Loss: `focal loss`
- Focal gamma: `1.5`
- Class-weight strategy: `sqrt_balanced_clipped`
- Class-weight overrides:
  - Asthma: `1.1`
  - Pneumonia: `1.75`
- Inference aggregation: `pneumonia_sensitive`
- Window overlap: `0.5`
- Max windows per recording: `5`

## Saved Results to Mention

### Validation Metrics from the Saved Strong-CNN Pneumonia-Focus Run

- Best validation accuracy: `0.7389`
- Best validation loss: `0.4388`
- Macro F1-score: `0.5132`
- Weighted F1-score: `0.7273`

Per-class validation F1:

- Asthma: `0.3846`
- COPD: `0.8714`
- Normal: `0.4333`
- Pneumonia: `0.3636`

### Test Recording-Level Metrics from the Saved Final Evaluation

- Accuracy: `0.8140`
- Macro precision: `0.5817`
- Macro recall: `0.5862`
- Macro F1-score: `0.5791`
- Weighted F1-score: `0.8075`

Per-class test F1:

- Asthma: `0.5625`
- COPD: `0.9243`
- Normal: `0.6190`
- Pneumonia: `0.2105`

### Interpretation of Results

These metrics suggest that:

- the model performs strongly for COPD
- the platform achieves good overall weighted performance at recording level
- window-level aggregation improves practical inference quality
- pneumonia remains the hardest class and still needs more balanced real-world data and stronger model tuning

This is a realistic and academically honest discussion point for the final report.

## Key Modules You Can Describe in Chapter 3

### Module 1: Data Collection and Dataset Preparation

- collect respiratory sound recordings from multiple datasets
- normalize labels
- remove duplicates
- organize files and metadata

### Module 2: Preprocessing and Feature Representation

- trim silence and normalize signal
- resample audio
- fit signals to fixed duration
- generate log-mel spectrograms
- compute delta and delta-delta channels

### Module 3: Model Training

- build training datasets
- apply augmentation
- train deep learning architectures
- compute evaluation metrics

### Module 4: Inference and Explainability

- run recording-level prediction
- aggregate window probabilities
- compute severity
- generate Grad-CAM style attention heatmaps

### Module 5: Web Platform

- upload respiratory audio
- visualize prediction results
- display explanations and report

### Module 6: Android Mobile Application

- record respiratory audio live
- upload audio to backend
- show mobile-friendly diagnostic output

## Suggested Report Positioning

The best way to present the project is:

- early prototype: FastAPI + baseline classifier + simple web interface
- final system: structured ML rebuild + Flask inference service + `PneumoScan` browser frontend + Android app

This shows clear project progression, which usually helps in major-project evaluation.

## Strong Points for Viva / Report Defense

- Complete end-to-end system, not only a model
- Includes both web and Android delivery channels
- Uses explainability through heatmap visualization
- Handles real recording workflow, not just offline CSV classification
- Uses patient-aware splitting to reduce leakage
- Contains practical engineering for deployment-style inference
- Includes minority-class improvement strategy for pneumonia

## Limitations to Mention Honestly

- The system is a screening aid, not a certified clinical diagnostic tool
- Dataset imbalance is still significant, especially for Asthma and Pneumonia
- Performance is strongest for COPD and weaker for Pneumonia
- More real-world hospital data and clinical validation are needed
- Environmental noise, recording quality, and device variability can affect prediction quality

## Good Candidate Objectives for the Report

- To design and develop an AI-assisted respiratory disease screening platform using lung sound recordings
- To preprocess respiratory audio and convert it into spectrogram-based model input
- To train and evaluate deep-learning models for multiclass respiratory condition prediction
- To provide explainable prediction outputs through heatmap-based visualization
- To integrate the trained model into both web and Android interfaces for accessible use

## Good Candidate Future Scope Points

- collect larger balanced datasets, especially for pneumonia and asthma
- add real-time noise reduction and signal quality scoring
- improve pneumonia sensitivity through new architectures or ensemble models
- integrate patient metadata such as age, symptoms, and oxygen saturation
- add PDF report export and patient history tracking
- deploy the backend securely on cloud infrastructure
- extend support to digital stethoscope and IoT-based capture devices

## Likely File References for Screenshots or Appendix

- root MVP backend: `app/main.py`
- root MVP feature extraction: `app/services/audio.py`
- root MVP heuristic inference: `app/services/inference.py`
- root static web UI: `static/index.html`
- final backend API: `respiratory_ai_rebuild/app/backend/main.py`
- final inference engine: `respiratory_ai_rebuild/src/resp_ai/inference/predictor.py`
- training pipeline: `respiratory_ai_rebuild/src/resp_ai/models/train.py`
- evaluation pipeline: `respiratory_ai_rebuild/src/resp_ai/models/evaluate.py`
- Android main activity: `respiratory_ai_android_app/app/src/main/java/com/respiratoryai/mobile/MainActivity.kt`
- Android view model: `respiratory_ai_android_app/app/src/main/java/com/respiratoryai/mobile/ui/MainViewModel.kt`

## What Is Still Needed From You

To convert these notes into the final college report, I still need:

- your exact college format and page-order rules
- student details, guide name, department, college name, and academic year
- whether the report should use first person or formal academic style
- whether you want the literature review written from cited papers
- whether you want the final report as Markdown, Word-ready text, or section-by-section content

